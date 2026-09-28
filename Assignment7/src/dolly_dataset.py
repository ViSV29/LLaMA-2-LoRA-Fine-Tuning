import argparse
from datasets import load_dataset, DatasetDict
from pathlib import Path

#Formatting Template
TEMPLATE = """### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
"""

#format each example
def formatExample(example):
    instruction = (example.get("instruction") or "").strip()
    context = (example.get("context") or "").strip()
    response = (example.get("response") or "").strip()

    #skip examples with missing responses
    if response == "":
        return None

    formattedText = TEMPLATE.format(
        instruction=instruction if instruction else "N/A",
        context=context if context else "N/A",
        response=response,
    )
    return {"text": formattedText}

def main(output_dir: str):
    print("Loading Dolly-15k from Hugging Face")
    raw_ds = load_dataset("databricks/databricks-dolly-15k")["train"]

    print("Cleaning and formatting")
    cleanedData = raw_ds.filter(lambda ex: ex["response"] and ex["response"].strip() != "")
    formatted = cleanedData.map(formatExample, remove_columns=raw_ds.column_names)

    #split 80/10/10
    formatted = formatted.shuffle(seed=42)
    n = len(formatted)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    ds_splits = DatasetDict({
        "train": formatted.select(range(0, n_train)),
        "validation": formatted.select(range(n_train, n_train + n_val)),
        "test": formatted.select(range(n_train + n_val, n)),
    })

    #save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ds_splits.save_to_disk(out_path.as_posix())

    #debugging
    print(f"Saved processed dataset to: {out_path}")
    print(f"   • Train: {n_train} samples")
    print(f"   • Validation: {n_val} samples")
    print(f"   • Test: {n_test} samples")
    print("\nExample sample:\n", ds_splits["train"][0]["text"][:400])

#CLI entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format Dolly-15k for instruction fine-tuning.")
    parser.add_argument("--output_dir", type=str, default="data/dolly", help="Where to save processed dataset")
    args = parser.parse_args()
    main(args.output_dir)
