import os
import json
import random
import time
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI

#config
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
LORA_ADAPTER_DIR = "/content/Assignment7/Assignment7/models/llama2_7b_dolly_lora_model"
OUT_DIR = "./mtbench_results"
MTBENCH_JSON = "/content/Assignment7/Assignment7/src/question.jsonl" 
EVAL_LIMIT = 80 
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0
USE_8BIT = True
BATCH_SIZE = 2
SEED = 1337
JUDGE_MODEL = "gpt-4o-mini"

#setup
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#api check
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable before running.")
client = OpenAI(api_key=api_key)

#load mt-bench eval dataset (saved local copy)
print("Loading MT-Bench dataset (local copy)…")
data = []
with open(MTBENCH_JSON, "r") as f:
    for line in f:
        try:
            data.append(json.loads(line))
        except Exception as e:
            print(f"Skipping line: {e}")

if EVAL_LIMIT:
    data = data[:EVAL_LIMIT]

prompts = [q["turns"][0] for q in data] 
instrs = [q.get("question_id", f"q{i}") for i, q in enumerate(data)]
print(f"Loaded {len(prompts)} single-turn questions from MT-Bench.")

#helper function
def formatPrompt(question):
    return f"### Instruction:\n{question}\n\n### Response:\n"

def batched_generate(model, tokenizer, prompts):
    outs = []
    model.eval()
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating"):
        batch = [formatPrompt(p) for p in prompts[i:i+BATCH_SIZE]]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **toks,
                do_sample=False,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
            )
        for j, g in enumerate(gen):
            outs.append(tokenizer.decode(g, skip_special_tokens=True))
    return outs

def openai_judge(question, ansA, ansB, model):
    system_prompt = (
        "You are a strict evaluator. Compare two answers to the SAME question.\n"
        "Choose exactly one: 'A' or 'B'. Use 'Tie' only if they are equally good.\n"
        "Judge based on correctness, helpfulness, and coherence."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Answer A (fine-tuned):\n{ansA}\n\n"
        f"Answer B (base):\n{ansB}\n\n"
        "Which answer is better? Reply with only one word: A, B, or Tie."
    )
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_prompt}],
                temperature=0,
            )
            resp = r.choices[0].message.content.strip()
            if resp.startswith("A"): return "A"
            if resp.startswith("B"): return "B"
            if resp.lower().startswith("tie"): return "Tie"
        except Exception as e:
            print(f"Judge error: {e}, retrying...")
            time.sleep(2)
    return "Tie"

#load models
print("Loading tokeniser")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
tok.padding_side = "left" 

print("Loading base model")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, device_map="auto", torch_dtype=torch.float16, load_in_8bit=USE_8BIT
)

print("Loading LoRA-tuned model")
ft_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, device_map="auto", torch_dtype=torch.float16, load_in_8bit=USE_8BIT
)
ft_model = PeftModel.from_pretrained(ft_model, LORA_ADAPTER_DIR)

#output
print("Generating outputs")
base_outs = batched_generate(base_model, tok, prompts)
ft_outs = batched_generate(ft_model, tok, prompts)

with open(f"{OUT_DIR}/base_outputs.json", "w") as f:
    json.dump([{"question": q, "output": o} for q, o in zip(prompts, base_outs)], f, indent=2)
with open(f"{OUT_DIR}/finetuned_outputs.json", "w") as f:
    json.dump([{"question": q, "output": o} for q, o in zip(prompts, ft_outs)], f, indent=2)

#comparison
print(f"Judging with {JUDGE_MODEL}...")
wins = ties = 0
decisions = []
for i, (q, a_ft, a_base) in enumerate(zip(prompts, ft_outs, base_outs)):
    decision = openai_judge(q, a_ft, a_base, JUDGE_MODEL)
    decisions.append(decision)
    if decision == "A": wins += 1
    elif decision == "Tie": ties += 1
    print(f"[{i+1}/{len(prompts)}] → {decision}")

raw_win_rate = wins / len(decisions)
summary = {"judge_model": JUDGE_MODEL, "samples": len(prompts),
            "raw_win_rate": raw_win_rate, "ties": ties}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n Evaluation complete!")
print(json.dumps(summary, indent=2))
