import torch
from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Computes the per-instance DPO loss:
      -log sigmoid[ beta * (logπθ(x⊕yw) - logπref(x⊕yw) - (logπθ(x⊕yl) - logπref(x⊕yl))) ]
    where logπ sums log-probs over all tokens in the sequence x⊕y.
    """
    def build_seq(response: str) -> str:
        s = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
            f"{response}"
        )
        # append exactly one EOS marker if defined
        return s + (tokenizer.eos_token or "")

    seq_pos = build_seq(response_chosen)
    seq_neg = build_seq(response_rejected)

    def sequence_logprob(model: torch.nn.Module, text: str) -> torch.Tensor:
        # tokenize (adds BOS/EOS if the tokenizer is set up that way)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        device = next(model.parameters()).device

        input_ids = enc["input_ids"].to(device)            # (1, L)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits                            # (1, L, V)

        # shift so that tokens t predict token t+1
        shift_logits = logits[:, :-1, :].contiguous()       # (1, L-1, V)
        shift_labels = input_ids[:, 1:].contiguous()        # (1, L-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)     # (1, L-1, V)
        token_logps = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, L-1)

        return token_logps.sum(dim=1).squeeze(0)             # scalar

    lp_th_pos = sequence_logprob(lm,     seq_pos)
    lp_th_neg = sequence_logprob(lm,     seq_neg)
    lp_rf_pos = sequence_logprob(lm_ref, seq_pos)
    lp_rf_neg = sequence_logprob(lm_ref, seq_neg)

    delta_th = lp_th_pos - lp_th_neg
    delta_rf = lp_rf_pos - lp_rf_neg

    diff = beta * (delta_th - delta_rf.to(delta_th.device))
    loss = -F.logsigmoid(diff)
    return loss
