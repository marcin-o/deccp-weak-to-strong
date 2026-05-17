from __future__ import annotations

import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(
    model_name: str,
    use_4bit: bool = True,
    trust_remote_code: bool = False,
):
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def build_chat_input(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    return prompt


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> str:
    input_text = build_chat_input(tokenizer, prompt)
    inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs.input_ids.shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def _get_resume_keys(existing_df: pd.DataFrame) -> set:
    if "prompt_id" in existing_df.columns:
        return set(existing_df["prompt_id"])

    return set()


def _is_processed(row: pd.Series, processed_keys: set) -> bool:
    return row["prompt_id"] in processed_keys


def run_model_to_csv(
    model_name: str,
    prompts_df: pd.DataFrame,
    output_file: str | Path,
    use_4bit: bool = True,
    max_new_tokens: int = 256,
    limit: int | None = None,
    trust_remote_code: bool = False,
) -> pd.DataFrame:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        processed_keys = _get_resume_keys(existing_df)
        results = existing_df.to_dict("records")
        print(f"Resuming from {len(results)} already saved rows.")
    else:
        processed_keys = set()
        results = []

    rows_to_process = [
        row for _, row in prompts_df.iterrows() if not _is_processed(row, processed_keys)
    ]

    if limit is not None:
        rows_to_process = rows_to_process[:limit]

    if not rows_to_process:
        print("No prompts left to process.")
        return pd.DataFrame(results)

    model, tokenizer = load_model(
        model_name=model_name,
        use_4bit=use_4bit,
        trust_remote_code=trust_remote_code,
    )

    try:
        for row in tqdm(rows_to_process, total=len(rows_to_process)):
            response = generate_answer(
                model=model,
                tokenizer=tokenizer,
                prompt=row["prompt"],
                max_new_tokens=max_new_tokens,
            )

            result = {
                "prompt_id": row["prompt_id"],
                "prompt": row["prompt"],
                "response": response,
            }

            results.append(result)
            pd.DataFrame(results).to_csv(output_path, index=False)

    finally:
        del model
        del tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pd.DataFrame(results)
