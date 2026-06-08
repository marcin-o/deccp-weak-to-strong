from __future__ import annotations

import gc
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def load_w2s_models(
    strong_model_id: str,
    weak_model_id: str,
    adapter_dir: str | Path,
    use_4bit: bool = True,
):
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {adapter_dir}. "
            "Fine-tune the weak model first (scripts/run_finetuning.py)."
        )

    quant = _bnb_config() if use_4bit else None
    common_kwargs = dict(device_map="auto", torch_dtype=torch.float16)
    if quant is not None:
        common_kwargs["quantization_config"] = quant

    print(f"-> Loading strong model {strong_model_id} ...")
    strong = AutoModelForCausalLM.from_pretrained(strong_model_id, **common_kwargs)
    strong.eval()

    print(f"-> Loading weak base model {weak_model_id} ...")
    weak_base = AutoModelForCausalLM.from_pretrained(weak_model_id, **common_kwargs)

    print(f"-> Attaching LoRA adapter from {adapter_dir} ...")
    weak = PeftModel.from_pretrained(weak_base, str(adapter_dir))
    weak.eval()

    tokenizer = AutoTokenizer.from_pretrained(strong_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    strong_vocab = strong.get_output_embeddings().weight.shape[0]
    weak_vocab = weak.get_output_embeddings().weight.shape[0]
    if strong_vocab != weak_vocab:
        print(
            f"WARNING: strong vocab ({strong_vocab}) != weak vocab ({weak_vocab}). "
            "Logits will be truncated to the common size."
        )

    return strong, weak, tokenizer


def _build_chat_input(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


@torch.no_grad()
def w2s_generate(
    strong,
    weak,
    tokenizer,
    prompt: str,
    alpha: float = 1.0,
    max_new_tokens: int = 256,
) -> str:
    device = strong.device
    input_text = _build_chat_input(tokenizer, prompt)
    input_ids = tokenizer([input_text], return_tensors="pt").input_ids.to(device)

    eos_id = tokenizer.eos_token_id

    strong_out = strong(input_ids, use_cache=True)
    expert_out = weak(input_ids, use_cache=True)
    with weak.disable_adapter():
        amateur_out = weak(input_ids, use_cache=True)

    strong_past = strong_out.past_key_values
    expert_past = expert_out.past_key_values
    amateur_past = amateur_out.past_key_values

    strong_logits = strong_out.logits[:, -1, :]
    expert_logits = expert_out.logits[:, -1, :]
    amateur_logits = amateur_out.logits[:, -1, :]

    generated: list[int] = []

    for _ in range(max_new_tokens):
        vocab = min(
            strong_logits.shape[-1],
            expert_logits.shape[-1],
            amateur_logits.shape[-1],
        )
        ls = F.log_softmax(strong_logits[:, :vocab].float(), dim=-1)
        le = F.log_softmax(expert_logits[:, :vocab].float(), dim=-1)
        la = F.log_softmax(amateur_logits[:, :vocab].float(), dim=-1)

        combined = ls + alpha * (le - la)
        next_token = combined.argmax(dim=-1)

        token_id = int(next_token.item())
        if token_id == eos_id:
            break
        generated.append(token_id)

        next_input = next_token.view(1, 1).to(device)

        strong_out = strong(
            next_input, past_key_values=strong_past, use_cache=True
        )
        expert_out = weak(
            next_input, past_key_values=expert_past, use_cache=True
        )
        with weak.disable_adapter():
            amateur_out = weak(
                next_input, past_key_values=amateur_past, use_cache=True
            )

        strong_past = strong_out.past_key_values
        expert_past = expert_out.past_key_values
        amateur_past = amateur_out.past_key_values

        strong_logits = strong_out.logits[:, -1, :]
        expert_logits = expert_out.logits[:, -1, :]
        amateur_logits = amateur_out.logits[:, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _resume_keys(existing_df: pd.DataFrame) -> set:
    if "prompt_id" in existing_df.columns:
        return set(existing_df["prompt_id"])
    return set()


def run_w2s_to_csv(
    strong_model_id: str,
    weak_model_id: str,
    adapter_dir: str | Path,
    prompts_df: pd.DataFrame,
    output_file: str | Path,
    alpha: float = 1.0,
    use_4bit: bool = True,
    max_new_tokens: int = 256,
    limit: int | None = None,
) -> pd.DataFrame:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        processed = _resume_keys(existing_df)
        results = existing_df.to_dict("records")
        print(f"Resuming from {len(results)} already saved rows.")
    else:
        processed = set()
        results = []

    rows_to_process = [
        row for _, row in prompts_df.iterrows() if row["prompt_id"] not in processed
    ]
    if limit is not None:
        rows_to_process = rows_to_process[:limit]

    if not rows_to_process:
        print("No prompts left to process.")
        return pd.DataFrame(results)

    strong, weak, tokenizer = load_w2s_models(
        strong_model_id=strong_model_id,
        weak_model_id=weak_model_id,
        adapter_dir=adapter_dir,
        use_4bit=use_4bit,
    )

    model_label = f"{strong_model_id}+w2s({weak_model_id}+LoRA,alpha={alpha})"

    try:
        for row in tqdm(rows_to_process, total=len(rows_to_process)):
            start = time.time()
            response = w2s_generate(
                strong=strong,
                weak=weak,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                alpha=alpha,
                max_new_tokens=max_new_tokens,
            )
            latency = time.time() - start

            results.append(
                {
                    "model": model_label,
                    "split": row.get("split", ""),
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "response": response,
                    "latency_seconds": latency,
                }
            )
            pd.DataFrame(results).to_csv(output_path, index=False)

    finally:
        del strong
        del weak
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(results)
