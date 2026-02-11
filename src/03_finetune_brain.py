import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import multiprocessing

# --- WINDOWS SAFE GUARD ---
def main():
    print("🟢 STARTING BRAIN TRAINING (LLM Fine-Tuning)...")

    # 1. SETUP PATHS
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "data"))
    MODELS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "models"))
    CSV_PATH = os.path.join(BASE_DATA_DIR, "ai_data.csv")

    # 2. LOAD & FORMAT DATA
    print(f"📂 Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print("❌ CRITICAL ERROR: ai_data.csv not found!")
        return

    df = pd.read_csv(CSV_PATH)
    
    # Format data for TinyLlama
    training_data = []
    for idx, row in df.iterrows():
        product = row['class']
        desc = row['description']
        price = row['price']
        
        # A. Description
        text_a = f"<|user|>Describe the {product}.<|assistant|>{desc}"
        training_data.append({"text": text_a})
        
        # B. Price
        text_b = f"<|user|>How much is the {product}?<|assistant|>The {product} costs ${price}."
        training_data.append({"text": text_b})

    dataset = Dataset.from_list(training_data)
    print(f"✅ Created {len(dataset)} training examples.")

    # 3. LOAD TINYLLAMA
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"⏳ Loading {model_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # 4. CONFIGURE LoRA
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16, 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    print("🧠 LoRA Adapter attached.")

    # 5. CONFIGURE TRAINING
    print("🏋️‍♂️ Training Started...")
    
    # Using 'max_length' as requested by your error log
    training_args = SFTConfig(
        output_dir="llm_checkpoints",
        dataset_text_field="text",
        max_length=128,              # <--- RENAMED FROM max_seq_length
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=30,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args
    )

    trainer.train()

    # 6. SAVE
    final_path = os.path.join(MODELS_DIR, "tinyllama_finetune")
    print(f"💾 Saving new brain to {final_path}...")
    model.save_pretrained(final_path)
    print("\n------------------------------------------------")
    print("🎉 BRAIN TRAINING COMPLETE!")
    print("✅ TinyLlama now knows about your specific products.")
    print("------------------------------------------------\n")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()