import os
import torch
import argparse
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

TEACHER_RESULTS_PATH = "results/baseline/judged/results_qwen_14b_judged.csv"

def get_model_dir_name(model_id):
    """Maps Hugging Face model ID to your local clean directory structure."""
    mapping = {
        "Qwen/Qwen2.5-0.5B-Instruct": "qwen_0_5b",
        "Qwen/Qwen2.5-1.5B-Instruct": "qwen_1_5b",
        "Qwen/Qwen2.5-14B-Instruct": "qwen_14b",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "deepseek_7b"
    }
    
    return mapping.get(model_id, model_id.split('/')[-1].lower().replace('.', '_').replace('-', '_'))

def load_training_data():
    """Loads only successful responses from the large model to use as training data."""
    print(f"-> Loading training data from: {TEACHER_RESULTS_PATH}")
    if not os.path.exists(TEACHER_RESULTS_PATH):
        raise FileNotFoundError(f"File not found: {TEACHER_RESULTS_PATH}")

    df = pd.read_csv(TEACHER_RESULTS_PATH)
    df_success = df[df['success'] == 1].copy()
    df_success = df_success[['prompt', 'response']]
    
    print(f"-> Found {len(df_success)} successful responses for fine-tuning!")
    return Dataset.from_pandas(df_success)

def format_chat_template(example, tokenizer):
    """Formats the prompt and response into a chat template recognized by the model."""
    messages = [
        {"role": "user", "content": example['prompt']},
        {"role": "assistant", "content": example['response']}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}

def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning tailored for Turing GPUs with dynamic models.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Hugging Face model ID to fine-tune."
    )
    args = parser.parse_args()
    
    model_id = args.model
    model_dir = get_model_dir_name(model_id)
    output_dir = f"results/steered/{model_dir}/adapters"
    
    print(f"=== Starting pipeline for: {model_id} ===")
    print(f"-> Target adapter directory: {output_dir}")

    dataset = load_training_data()

    print(f"-> Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=["prompt", "response"]
    )

    print("-> Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("-> Configuring SFTTrainer (Optimized for Turing Architecture)...")
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=8,       
        learning_rate=2e-4,
        logging_steps=5,
        max_steps=50,
        optim="paged_adamw_8bit",
        fp16=False,                          
        bf16=False,                          
        gradient_checkpointing=True,         
        report_to="none",
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args
    )

    print("-> Starting fine-tuning! Safe execution mode initiated...")
    trainer.train()
    
    print(f"-> Saving fine-tuned model adapters to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"=== FINETUNING FOR {model_id} COMPLETED SUCCESSFULLY! ===")

if __name__ == "__main__":
    main()