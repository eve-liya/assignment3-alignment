import re
from typing import Any

def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Parse model output into 'A'/'B'/'C'/'D', else None.
    """
    text = model_output.strip().upper()
    m = re.match(r'^([A-D])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'THE CORRECT ANSWER IS\s*([A-D])', text)
    if m:
        return m.group(1)
    m = re.search(r'ANSWER[:\s]*IS?\s*([A-D])', text)
    if m:
        return m.group(1)
    m = re.search(r'\(([A-D])\)', text)
    if m:
        return m.group(1)
    return None

def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Parse GSM8K model output by taking the last numeric value found.
    """
    nums = re.findall(r'[-+]?\d*\.?\d+', model_output)
    if not nums:
        return None
    return nums[-1]
