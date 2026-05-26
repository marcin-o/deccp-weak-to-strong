import os
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "results/finetuned_weak_model"
PROMPTS_PATH = "data/deccp_prompts.csv"
OUTPUT_PATH = "results/finetuned_weak_model_responses.csv"

def main():
    print(f"-> Loading tokenizer for {BASE_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"-> Loading base model in 4-bit with Float16 weights...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print(f"-> Loading and applying LoRA adapters from {ADAPTER_DIR}...")
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()  

    print(f"-> Reading prompts from {PROMPTS_PATH}...")
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
            "model": f"{BASE_MODEL_ID}+LoRA",
            "split": split_name,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "response": response_text,
            "latency_seconds": latency
        })
        
        print(f"[{idx + 1}/{len(prompts_df)}] Generated response for prompt_id {prompt_id} in {latency:.2f}s")

    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    
    output_df = pd.DataFrame(generated_results)
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"SUCCESS: Saved {len(output_df)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()