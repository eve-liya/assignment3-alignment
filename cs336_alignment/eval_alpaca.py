#!/usr/bin/env python3
import os
import json
import time
import argparse
import logging
import random
from typing import Any, Dict, List

# Prevent vLLM usage reporting and MKL threading errors
os.environ["VLLM_DISABLE_USAGE_REPORTING"] = "1"
os.environ["MKL_THREADING_LAYER"]    = "GNU"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

from vllm import LLM, SamplingParams

def load_alpaca_eval(path: str) -> List[Dict[str, Any]]:
    """
    Each line of the JSONL should be an object with at least:
      - "instruction": str
      - "dataset":     str
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure required keys
            instr  = obj.get("instruction")
            ds      = obj.get("dataset", "alpaca_eval")
            if instr is None:
                continue
            examples.append({"instruction": instr, "dataset": ds})
    return examples

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file",    required=True,
                        help="Path to AlpacaEval JSONL (one JSON per line).")
    parser.add_argument("--model-path",   required=True,
                        help="Path to Qwen2.5-0.5B model directory.")
    parser.add_argument("--generator-name", default="qwen2.5-0.5b",
                        help="String to tag as the 'generator' field.")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--output-file",  default="alpaca_zero_shot.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 1) Load examples
    examples = load_alpaca_eval(args.data_file)
    n = len(examples)
    if n == 0:
        logging.error("No examples loaded; check --data-file")
        return
    logging.info(f"Loaded {n} AlpacaEval instructions from {args.data_file}")

    # 2) Prepare vLLM
    llm = LLM(model=args.model_path)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
        stop=["\n"]
    )

    # 3) Generate zero-shot outputs in batches
    outputs: List[Dict[str, Any]] = []
    t0 = time.time()
    for i in range(0, n, args.batch_size):
        batch = examples[i : i + args.batch_size]
        instrs = [ex["instruction"] for ex in batch]
        results = llm.generate(instrs, sampling)
        for ex, res in zip(batch, results):
            text = res.outputs[0].text.rstrip()
            outputs.append({
                "instruction": ex["instruction"],
                "output":      text,
                "generator":   args.generator_name,
                "dataset":     ex["dataset"],
            })
    t1 = time.time()
    throughput = n / (t1 - t0)

    # 4) Serialize to a JSON array
    with open(args.output_file, "w", encoding="utf-8") as fout:
        json.dump(outputs, fout, indent=2, ensure_ascii=False)
    logging.info(f"Wrote {n} predictions to {args.output_file}")

    # 5) Print summary for (b)
    print("\n=== AlpacaEval ===")
    print(f"Total examples:   {n}")
    print(f"Throughput:       {throughput:.2f} examples/sec")
    print(f"Output file:      {args.output_file}\n")

    print("Sample prediction:")
    sample = random.choice(outputs)
    print(json.dumps(sample, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
