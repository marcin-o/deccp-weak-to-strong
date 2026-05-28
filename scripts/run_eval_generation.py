import os
import time
import torch
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROMPTS_PATH = "data/deccp_prompts.csv"

def get_model_dir_name(model_id):
    """Maps Hugging Face model ID to your local clean directory structure."""
    mapping = {
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen_0_5b",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen_1_5b",
        "Qwen/Qwen2.5-14B-Instruct": "qwen_14b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek_7b"
    }
    return mapping.get(model_id, model_id.split('/')[-1].lower().replace('.', '_').replace('-', '_'))

def main():
    parser = argparse.ArgumentParser(description="Generate responses from fine-tuned LoRA adapters dynamically.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID that was fine-tuned."
    )
    args = parser.parse_args()

    model_id = args.model
    model_dir = get_model_dir_name(model_id)
    adapter_dir = f"results/steered/{model_dir}/adapters"
    output_path = f"results/steered/{model_dir}/raw_responses.csv"

    print(f"=== Starting Evaluation Generation for: {model_id} ===")
    print(f"-> Reading adapters from: {adapter_dir}")
    print(f"-> Target output CSV: {output_path}")

    print(f"-> Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"-> Loading base model in 4-bit with Float16 weights...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print(f"-> Applying LoRA adapters from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval() 

    if not os.path.exists(PROMPTS_PATH):
        raise FileNotFoundError(f"Prompts file not found at {PROMPTS_PATH}")
    
    prompts_df = pd.read_csv(PROMPTS_PATH)
    generated_results = []

    print("-> Starting text generation loop...")
    for idx, row in prompts_df.iterrows():
        prompt_text = row['prompt']
        prompt_id = row['prompt_id']
        split_name = row['split']
        
        messages = [{"role": "user", "content": prompt_text}]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer([text_input], return_tensors="pt").to("cuda")
        
        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False  
            )
        latency = time.time() - start_time
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        generated_results.append({
            "model": f"{model_id}+LoRA",
            "split": split_name,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "response": response_text,
            "latency_seconds": latency
        })
        
        print(f"[{idx + 1}/{len(prompts_df)}] Generated prompt_id {prompt_id} in {latency:.2f}s")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_df = pd.DataFrame(generated_results)
    output_df.to_csv(output_path, index=False)
    print(f"=== SUCCESS: Saved responses to {output_path} ===")

if __name__ == "__main__":
    main()