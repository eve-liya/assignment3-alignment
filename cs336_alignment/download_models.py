from transformers import AutoModelForCausalLM, AutoTokenizer

for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-3B-Instruct"):
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model.save_pretrained(f"../{name.replace('/', '_')}")
    tokenizer.save_pretrained(f"../{name.replace('/', '_')}")
