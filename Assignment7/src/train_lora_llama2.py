import os
import argparse
from typing import Any, Dict
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import pandas as pd
import matplotlib.pyplot as plt

BASE_MODEL = "meta-llama/Llama-2-7b-hf"
DATA_DIR = "data/dolly"
OUTPUT_DIR = "models/llama2-7b-dolly-lora"

#tokenisation
def tokeniseFN(example: Dict[str, Any], tokenizer, max_len: int = 1024):
    text = example["text"]
    token = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    token["labels"] = token["input_ids"].copy()
    return token

def main(
    base_model: str = BASE_MODEL,
    data_dir: str = DATA_DIR,
    output_dir: str = OUTPUT_DIR,
    epochs: int = 3,
    per_device_train_batch_size: int = 2,
    grad_accum: int = 8,
    lr: float = 2e-4,
    max_len: int = 1024,
    load_4bit: bool = True,
):

    os.makedirs(output_dir, exist_ok=True)

    print("Loading tokeniser")
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset")
    ds = load_from_disk(data_dir)
    trainData, valData = ds["train"], ds["validation"]

    print("Tokenising")
    trainData = trainData.map(
        lambda x: tokeniseFN(x, tokenizer, max_len),
        remove_columns=trainData.column_names,
        num_proc=os.cpu_count(),
    )
    valData = valData.map(
        lambda x: tokeniseFN(x, tokenizer, max_len),
        remove_columns=valData.column_names,
        num_proc=os.cpu_count(),
    )

    print("Load base model")
    if load_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_4bit=True,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    print("Configuring LoRA adapter")
    loraConfig = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, loraConfig)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",          # smoother schedule
        warmup_ratio=0.03,
        max_grad_norm=1.0,                   # stability
        evaluation_strategy="epoch",         
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        bf16=torch.cuda.is_available(),
        weight_decay=0.0,
        report_to="none",
        optim="paged_adamw_8bit",            # faster + memory-efficient optimiser
        load_best_model_at_end=True,         
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainData,
        eval_dataset=valData,
        data_collator=data_collator,
    )

    print("Starting LoRA fine-tuning")
    trainer.train()

    print("Saving model")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # plot training curves
    metrics = pd.DataFrame(trainer.state.log_history)
    metrics.to_csv(f"{output_dir}/training_log.csv", index=False)

    trainLoss = metrics[metrics["loss"].notna()]
    evalLoss = metrics[metrics["eval_loss"].notna()] 
    if not trainLoss.empty and not evalLoss.empty:
        plt.figure(figsize=(7, 4))
        plt.plot(trainLoss["step"], trainLoss["loss"], label="Train Loss")
        plt.plot(evalLoss["step"], evalLoss["eval_loss"], label="Validation Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss (LoRA Fine-Tuning)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_curve.png")
        print(f"Saved loss curve to {output_dir}/loss_curve.png")

    print(f"Training complete! Model saved to {output_dir}")

# CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimised LoRA fine-tuning for Llama-2-7B on Dolly-15K")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    main(
        base_model=args.base_model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        max_len=args.max_len,
        load_4bit=not args.no_4bit,
    )
