import os, json, random, time
import numpy as np
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm 
import transformers
transformers.logging.set_verbosity_error() 


#config for colab
BASE_MODEL_ID = "meta-llama/Llama-2-7b-hf"
LORA_ADAPTER_DIR = "/content/Assignment7/Assignment7/models/llama2_7b_dolly_lora_model"
OUT_DIR = "./alpacaeval_mini_results"
EVAL_LIMIT = 100                # set to None for full set
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

#check api_key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set your OPENAI_API_KEY environment variable before running.")
client = OpenAI(api_key=api_key)

#helper fucntions
def formatPrompt(instruction, ctx=""):
    return (
        "### Instruction:\n" + instruction.strip() + "\n\n"
        "### Context:\n" + (ctx.strip() if ctx else "N/A") + "\n\n"
        "### Response:\n"
    )

def batchedGenerate(model, tokenizer, prompts):
    outs = []
    model.eval()
    for i in tqdm(range(0, len(prompts), BATCH_SIZE), desc="Generating", ncols=80):
        batch = prompts[i:i+BATCH_SIZE]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **toks,
                do_sample=False,
                temperature=TEMPERATURE,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id
            )
        for j, g in enumerate(gen):
            full = tokenizer.decode(g, skip_special_tokens=True)
            outs.append(full[len(batch[j]):].strip())
    return outs

def openai_judge(question, ansA, ansB, model):
    system_prompt = (
        "You are a strict evaluator. Compare two answers to the SAME instruction.\n"
        "Choose exactly one: 'A' or 'B'. Use 'Tie' only if they are equally good.\n"
        "Judge based on correctness, helpfulness, and coherence."
    )
    user_prompt = (
        f"Instruction:\n{question}\n\n"
        f"Answer A (fine-tuned):\n{ansA}\n\n"
        f"Answer B (base):\n{ansB}\n\n"
        "Which answer is better? Reply with only one word: A, B, or Tie."
    )
    for attempt in range(5):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )
            resp = r.choices[0].message.content.strip()
            if resp.startswith("A"): return "A"
            if resp.startswith("B"): return "B"
            if resp.lower().startswith("tie"): return "Tie"
            return "Tie"
        except Exception as e:
            print(f"Judge error: {e}, retrying...")
            time.sleep(2)
    return "Tie"

#load dataset -> downloaded the data locally first as url kept throwin error
LOCAL_DATASET_PATH = "/content/Assignment7/Assignment7/src/alpaca_eval.json"

print("Loading AlpacaEval dataset")
with open(LOCAL_DATASET_PATH, "r") as f:
    data = json.load(f)
ds = Dataset.from_list(data)
print(f"Loaded {len(ds)} examples from local AlpacaEval dataset.")

if EVAL_LIMIT:
    ds = ds.select(range(min(EVAL_LIMIT, len(ds))))
prompts = [formatPrompt(ex["instruction"], ex.get("input", "")) for ex in ds]
instructions = [ex["instruction"] for ex in ds]

#load models
print("Loading tokenizer")
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, use_fast=False)
tok.pad_token = tok.eos_token
tok.padding_side = "left"  #fixes right-padding warnin

print("Loading BASE model")
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
base_outs = batchedGenerate(base_model, tok, prompts)
ft_outs = batchedGenerate(ft_model, tok, prompts)

# Save outputs
with open(f"{OUT_DIR}/base_outputs.json", "w") as f:
    json.dump([{"instruction": i, "output": o} for i, o in zip(instructions, base_outs)], f, indent=2)
with open(f"{OUT_DIR}/finetuned_outputs.json", "w") as f:
    json.dump([{"instruction": i, "output": o} for i, o in zip(instructions, ft_outs)], f, indent=2)

#judge output
print(f"Judging with {JUDGE_MODEL}...")
wins = ties = 0
decisions = []

for i, (q, a_ft, a_base) in enumerate(
    tqdm(zip(instructions, ft_outs, base_outs), total=len(instructions), desc="Judging", ncols=80)
):
    decision = openai_judge(q, a_ft, a_base, JUDGE_MODEL)
    decisions.append(decision)
    if decision == "A": wins += 1
    elif decision == "Tie": ties += 1

raw_win_rate = wins / len(decisions)
print(f"\nRaw win rate (FT over Base): {raw_win_rate:.3f} | ties: {ties}/{len(decisions)}")

#length controlled win rate
lens = np.array([len(a.split()) for a in ft_outs]) - np.array([len(b.split()) for b in base_outs])
y = np.array([1 if d == "A" else 0 for d in decisions if d in ["A", "B"]])
lens = lens[:len(y)].reshape(-1, 1)
if len(y) > 5:
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(lens, y)
    lc_wr = float(clf.predict_proba(np.zeros((len(y), 1)))[:, 1].mean())
else:
    lc_wr = None

#save everything
summary = {
    "judge_model": JUDGE_MODEL,
    "samples": len(ds),
    "raw_win_rate": raw_win_rate,
    "length_controlled_win_rate": lc_wr,
}
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nEvaluation complete!")
print(json.dumps(summary, indent=2))
