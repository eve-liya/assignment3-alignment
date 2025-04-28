#!/usr/bin/env python3
import os
import csv
import json
import time
import argparse
import random
import torch
from typing import Any, List, Dict
from vllm import LLM, SamplingParams
from response_parsing import run_parse_mmlu_response

def load_mmlu_csv(csv_path: str, subject: str) -> List[Dict[str, Any]]:
    examples = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6: 
                continue
            q, *opts, ans = row
            examples.append({
                "subject": subject,
                "question": q,
                "options": opts[:4],
                "answer": ans.strip().upper()
            })
    return examples

def load_all_examples(base_dir: str) -> List[Dict[str, Any]]:
    examples = []
    for fn in sorted(os.listdir(base_dir)):
        if not fn.endswith(".csv"):
            continue
        subject = fn.split("_")[0]
        path = os.path.join(base_dir, fn)
        print(f"Loading {path} as subject={subject}")
        examples.extend(load_mmlu_csv(path, subject))
    return examples

def format_prompt(example: Dict[str, Any]) -> str:
    return (
        f"Answer the following multiple choice question about {example['subject']}. "
        "Respond with a single sentence of the form \"The correct answer is _\".\n\n"
        f"Question: {example['question']}\n"
        f"A. {example['options'][0]}\n"
        f"B. {example['options'][1]}\n"
        f"C. {example['options'][2]}\n"
        f"D. {example['options'][3]}\n"
        "Answer: "
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    required=True, help="Folder of MMLU CSVs")
    p.add_argument("--model-path",  required=True, help="Path to Qwen model")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--output-file", default="mmlu_dev_results.json")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    torch.cuda.empty_cache()

    # 1) Load and promptify
    examples = load_all_examples(args.data_dir)
    prompts  = [format_prompt(ex) for ex in examples]
    golds    = [ex["answer"] for ex in examples]
    print(f"Loaded {len(examples)} examples.")

    # 2) Prepare LLM
    llm = LLM(model=args.model_path)
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=16, stop=["\n"]
    )

    # 3) Generate in batches & time it
    t0, outputs = time.time(), []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        res = llm.generate(batch, sampling)
        outputs.extend(res)
        torch.cuda.empty_cache()
    t1 = time.time()
    throughput = len(prompts) / (t1 - t0)

    # 4) Parse and score
    records, num_correct, num_fail = [], 0, 0
    for ex, out, gold in zip(examples, outputs, golds):
        raw = out.outputs[0].text
        pred = run_parse_mmlu_response(ex, raw)
        correct = (pred == gold)
        if correct:   num_correct += 1
        if pred is None: num_fail += 1
        records.append({
            "subject": ex["subject"],
            "question": ex["question"],
            "options": ex["options"],
            "gold": gold,
            "raw_output": raw,
            "pred": pred,
            "correct": correct,
        })
    accuracy = num_correct / len(examples)

    # 5) Dump results
    stats = {
        "num_examples": len(examples),
        "accuracy": accuracy,
        "parse_failures": num_fail,
        "throughput_examples_per_sec": throughput,
    }
    out_data = {"stats": stats, "records": records}
    with open(args.output_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"→ Results saved to {args.output_file}\n")

    # 6) Print summary
    print("=== MMLU Eval Summary ===")
    print(f"Examples:        {len(examples)}")
    print(f"Accuracy:        {accuracy*100:.2f}%")
    print(f"Parse failures:  {num_fail}")
    print(f"Throughput:      {throughput:.2f} ex/sec\n")

    # 7) Show sample failures & errors
    if num_fail > 0:
        print("Sample parse failures:")
        for rec in records:
            if rec["pred"] is None:
                print(" RAW:", rec["raw_output"])
                break
    print("\n10 random incorrect examples:")
    wrong = [r for r in records if not r["correct"] and r["pred"] is not None]
    for i, r in enumerate(random.sample(wrong, min(10, len(wrong))), 1):
        print(f"{i}. Q: {r['question']}")
        print(f"   opts: {r['options']}")
        print(f"   pred={r['pred']} raw={r['raw_output']!r} gold={r['gold']}\n")

if __name__ == "__main__":
    main()
