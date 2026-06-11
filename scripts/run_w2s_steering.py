import os
import time
import torch
import argparse
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROMPTS_PATH = "data/deccp_prompts.csv"

def get_model_dir_name(model_id):
    """Maps Hugging Face model ID to your local clean directory structure."""
    mapping = {
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen_0_5b",
        "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit": "qwen_0_5b",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen_1_5b",
        "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit": "qwen_1_5b",
        "Qwen/Qwen2.5-7B-Instruct": "qwen_7b",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit": "qwen_7b",
        "Qwen/Qwen2.5-14B-Instruct": "qwen_14b",
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit": "qwen_14b"
    }
    return mapping.get(model_id, model_id.split('/')[-1].lower().replace('.', '_').replace('-', '_'))

def parse_args():
    parser = argparse.ArgumentParser(description="Weak-to-Strong Steering via Logit Algebra.")
    parser.add_argument(
        "--strong_model", 
        type=str, 
        default="Qwen/Qwen2.5-14B-Instruct",
        help="The large target model to be steered."
    )
    parser.add_argument(
        "--weak_model", 
        type=str, 
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="The base weak model identifier."
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=1.0,
        help="Amplification factor for steering. Higher = stronger shift."
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum tokens to generate per prompt."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamiczne ustalenie ścieżek
    weak_dir = get_model_dir_name(args.weak_model)
    strong_dir = get_model_dir_name(args.strong_model)
    
    adapter_dir = f"results/steered/{weak_dir}/adapters"
    output_path = f"results/steered/{strong_dir}/raw_responses.csv"
    
    print(f"=== Starting Weak-to-Strong Steering ===")
    print(f"-> Strong Model (Target): {args.strong_model}")
    print(f"-> Weak Model (Base):     {args.weak_model}")
    print(f"-> LoRA Adapters (Expert): {adapter_dir}")
    print(f"-> Amplification (Alpha):  {args.alpha}")
    print(f"-> Target Output CSV:     {output_path}")

    # --- MECHANIZM AUTOMATYCZNEGO WZNAWIANIA (AUTO-RESUME) ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    start_idx = 0
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            start_idx = len(existing_df)
            print(f"-> Wykryto istniejący plik z {start_idx} wynikami. Wznawiam bezpiecznie od indeksu {start_idx}...")
        except Exception:
            print(f"-> Plik istnieje, ale jest pusty lub uszkodzony. Zaczynam od zera.")
            os.remove(output_path)
    # ---------------------------------------------------------

    print("-> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.strong_model)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    print(f"-> Loading strong model: {args.strong_model}...")
    model_strong = AutoModelForCausalLM.from_pretrained(
        args.strong_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print(f"-> Loading weak model with adapters: {args.weak_model}...")
    base_weak = AutoModelForCausalLM.from_pretrained(
        args.weak_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model_weak = PeftModel.from_pretrained(base_weak, adapter_dir)
    model_weak.eval()
    model_strong.eval()

    if not os.path.exists(PROMPTS_PATH):
        raise FileNotFoundError(f"Prompts file not found at {PROMPTS_PATH}")
    prompts_df = pd.read_csv(PROMPTS_PATH)

    print("-> Starting steering generation loop...")
    for idx, row in prompts_df.iterrows():
        # --- POMIŃ PROMPTY, KTÓRE SĄ JUŻ W PLIKU CSV ---
        if idx < start_idx:
            continue
        # -----------------------------------------------
        
        prompt_text = row['prompt']
        prompt_id = row['prompt_id']
        split_name = row['split']
        
        messages = [{"role": "user", "content": prompt_text}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_ids = tokenizer([text_input], return_tensors="pt").input_ids.to("cuda")
        initial_length = input_ids.shape[1]
        
        start_time = time.time()
        
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                logits_strong = model_strong(input_ids).logits[:, -1, :]
                log_probs_strong = F.log_softmax(logits_strong, dim=-1)
                
                logits_weak_expert = model_weak(input_ids).logits[:, -1, :]
                log_probs_weak_expert = F.log_softmax(logits_weak_expert, dim=-1)
                
                with model_weak.disable_adapter():
                    logits_weak_base = model_weak(input_ids).logits[:, -1, :]
                    log_probs_weak_base = F.log_softmax(logits_weak_base, dim=-1)
            
            min_vocab = min(log_probs_strong.shape[-1], log_probs_weak_expert.shape[-1])
            log_probs_strong = log_probs_strong[:, :min_vocab]
            log_probs_weak_expert = log_probs_weak_expert[:, :min_vocab]
            log_probs_weak_base = log_probs_weak_base[:, :min_vocab]
            
            steered_log_probs = log_probs_strong + args.alpha * (log_probs_weak_expert - log_probs_weak_base)
            next_token = torch.argmax(steered_log_probs, dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
        latency = time.time() - start_time
        
        generated_ids = input_ids[:, initial_length:]
        response_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        
        # Pojedynczy rekord (wiersz)
        single_result = {
            "model": f"{args.strong_model}+W2S-Steered",
            "split": split_name,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "response": response_text,
            "latency_seconds": latency
        }
        
        # Zapis na bieżąco do CSV: mode='a' dopisuje wiersz, header=True tylko jeśli plik powstaje od zera
        row_df = pd.DataFrame([single_result])
        row_df.to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
        
        print(f"[{idx + 1}/{len(prompts_df)}] Steered prompt_id {prompt_id} in {latency:.2f}s (Saved to CSV)")

    print(f"=== SUCCESS: All responses generated and streamed to {output_path} ===")

if __name__ == "__main__":
    main()