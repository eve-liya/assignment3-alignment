#!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
import logging
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Disable vLLM usage reporting and force MKL threading layer
os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"
os.environ["MKL_THREADING_LAYER"]    = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

def load_sst_csv(path: str) -> list[str]:
    """
    Load SimpleSafetyTests prompts from a CSV with a column 'prompts_final'
    (or 'instruction' as fallback).
    Returns list of prompt strings.
    """
    prompts = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("prompts_final") or row.get("instruction")
            if p and p.strip():
                prompts.append(p.strip())
    return prompts

def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--csv-file",    required=True,
                   help="Path to simple_safety_tests.csv")
    p.add_argument("--model-path",  required=True,
                   help="Path to Qwen2.5-0.5B model directory")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--output-file", default="sst_zero_shot.jsonl")
    args = p.parse_args()

    # Load prompts
    prompts = load_sst_csv(args.csv_file)
    n = len(prompts)
    if n == 0:
        logging.error("No prompts loaded; check --csv-file")
        return
    logging.info(f"Loaded {n} SimpleSafetyTests prompts")

    # Initialize vLLM
    llm = LLM(model=args.model_path)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["\n"],
    )

    # Generate and time
    start = time.time()
    outputs = []
    for i in tqdm(range(0, n, args.batch_size), desc="Generating"):
        batch = prompts[i : i + args.batch_size]
        results = llm.generate(batch, sampling)
        outputs.extend(results)
    elapsed = time.time() - start
    throughput = n / elapsed

    # Write JSON-lines
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for prompt, out in zip(prompts, outputs):
            fout.write(json.dumps({
                "prompts_final": prompt,
                "output":        out.outputs[0].text.rstrip()
            }) + "\n")

    # Summary for (b)
    print("\n=== SimpleSafetyTests ===")
    print(f"Total examples:   {n}")
    print(f"Throughput:       {throughput:.2f} examples/sec")
    print(f"Outputs written:  {args.output_file}\n")

    # Show a sample
    print("Sample output:")
    print(outputs[0].outputs[0].text.strip())

if __name__ == "__main__":
    main()
