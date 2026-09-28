import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_lora_model(base_model_name: str, peft_path: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(peft_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, peft_path)
    model.eval()
    return model, tokenizer

def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)
