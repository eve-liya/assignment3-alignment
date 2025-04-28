import gzip
import json
import random
from pathlib import Path
from typing import List, Dict, Any
import os

def parse(convo: str) -> List[tuple[str,str]]:
    """
    Turn a single string like
       "Human: What is 2+2?\nAssistant: It is 4.\n"
    into a list of (speaker, text) tuples.
    """
    msgs: List[tuple[str,str]] = []
    for line in convo.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Human:"):
            msgs.append(("human", line[len("Human:"):].strip()))
        elif line.startswith("Assistant:"):
            msgs.append(("assistant", line[len("Assistant:"):].strip()))
    return msgs

def load_hh_dataset(base_dir: str) -> List[Dict[str, Any]]:
    """
    Load and preprocess the Anthropic HH preference data from base_dir.
    Expects these files in base_dir:
      - harmless-base.jsonl.gz
      - helpful-online.jsonl.gz
      - helpful-base.jsonl.gz
      - helpful-rejection-sampled.jsonl.gz

    Returns a list of dicts, each with keys:
      * instruction: str
      * chosen: str   (the assistant’s preferred reply)
      * rejected: str (the assistant’s dispreferred reply)
      * source: str   (which file it came from)
    """
    splits = [
        "harmless-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz",
    ]
    data: List[Dict[str, Any]] = []

    for fname in splits:
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            print(f"⚠️  File not found: {path}")
            continue
        print(f"Loading {path}...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                chosen_raw   = entry.get("chosen", "")
                rejected_raw = entry.get("rejected", "")

                chosen_msgs   = parse(chosen_raw)
                rejected_msgs = parse(rejected_raw)

                # only keep exact 2-message (human→assistant) single turns
                if len(chosen_msgs)==2 and len(rejected_msgs)==2:
                    (h_chosen, instr), (a_chosen, good) = chosen_msgs
                    (h_rej, _),      (a_rej, bad)   = rejected_msgs

                    if h_chosen=="human" and a_chosen=="assistant" and h_rej=="human" and a_rej=="assistant":
                        data.append({
                            "instruction": instr,
                            "chosen":      good,
                            "rejected":    bad,
                            "source":      fname,
                        })

    print(f"Total examples loaded: {len(data)}")
    return data

if __name__ == "__main__":
    import random
    hh = load_hh_dataset("hh_rlhf")
    # separate by harmless vs. helpful
    harmless = [ex for ex in hh if ex["source"].startswith("harmless")]
    helpful  = [ex for ex in hh if ex["source"].startswith("helpful")]

    print(f"Harmless examples: {len(harmless)}; Helpful examples: {len(helpful)}\n")

    print("=== 3 Random Harmless ===")
    for ex in random.sample(harmless, 3):
        print("\n— Instruction:", ex["instruction"])
        print("  Chosen   :", ex["chosen"])
        print("  Rejected :", ex["rejected"])

    print("\n=== 3 Random Helpful ===")
    for ex in random.sample(helpful, 3):
        print("\n— Instruction:", ex["instruction"])
        print("  Chosen   :", ex["chosen"])
        print("  Rejected :", ex["rejected"])

