#!/usr/bin/env python3
import os
import json
import time
import argparse
import random
import logging
from typing import Any, Dict, List, Optional

from vllm import LLM, SamplingParams
from response_parsing import run_parse_gsm8k_response

def load_gsm8k_local(path: str) -> List[Dict[str, Any]]:
    """Load a local GSM8K JSONL file with fields 'question' and 'answer'."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            if 'question' in obj and 'answer' in obj:
                examples.append({
                    'question': obj['question'],
                    'answer': obj['answer']
                })
    return examples

def load_gsm8k(path: Optional[str]) -> List[Dict[str, Any]]:
    if path and os.path.exists(path):
        return load_gsm8k_local(path)
    # Fallback to HuggingFace if no local file provided
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    return [{'question': ex['question'], 'answer': ex['answer']} for ex in ds]

def format_prompt(question: str) -> str:
    return f"{question}\nAnswer:"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file",    type=str, help="Path to GSM8K JSONL test file")
    parser.add_argument("--model-path",   required=True, help="Path to Qwen2.5-0.5B model")
    parser.add_argument("--batch-size",   type=int, default=8)
    parser.add_argument("--output-file",  default="gsm8k_zero_results.json")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    logging.basicConfig(level=logging.INFO)

    # Load examples
    examples = load_gsm8k(args.data_file)
    questions = [ex['question'] for ex in examples]
    golds     = [ex['answer'].strip()           for ex in examples]
    prompts   = [format_prompt(q) for q in questions]
    n = len(prompts)
    logging.info(f"Loaded {n} examples.")

    # Initialize vLLM
    llm = LLM(model=args.model_path)
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        stop=["\n"]
    )

    # Generate with timing
    t0, outputs = time.time(), []
    for i in range(0, n, args.batch_size):
        batch = prompts[i : i + args.batch_size]
        outputs.extend(llm.generate(batch, sampling))
    t1 = time.time()
    throughput = n / (t1 - t0)

    # Parse and evaluate
    records, num_correct, num_fail = [], 0, 0
    for ex, out, gold in zip(examples, outputs, golds):
        raw = out.outputs[0].text
        pred = run_parse_gsm8k_response(raw)
        ok   = (pred == gold)
        if ok:   num_correct += 1
        if pred is None: num_fail += 1
        records.append({
            "question":      ex['question'],
            "gold_answer":   gold,
            "model_output":  raw,
            "parsed_answer": pred,
            "correct":       ok
        })

    accuracy = num_correct / n

    # Assemble stats
    stats = {
        "num_examples":       n,
        "accuracy":           accuracy,
        "num_correct":        num_correct,
        "parse_failures":     num_fail,
        "throughput_ex_per_s": throughput
    }

    # Output to JSON
    with open(args.output_file, "w") as f:
        json.dump({"stats": stats, "records": records}, f, indent=2)
    logging.info(f"Results written to {args.output_file}")

    # Print summary (c)-(e)
    print("\n=== GSM8K ===")
    print(f"Total examples:   {n}")
    print(f"Accuracy:         {accuracy*100:.2f}% ({num_correct}/{n})")
    print(f"Parse failures:   {num_fail}")
    print(f"Throughput:       {throughput:.2f} examples/sec\n")

    if num_fail > 0:
        print("Sample parse failure (raw output):")
        for rec in records:
            if rec["parsed_answer"] is None:
                print(" ", rec["model_output"])
                break

if __name__ == "__main__":
    main()