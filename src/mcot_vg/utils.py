import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


INVALID_ANSWERS = {
    "a okvqa", "okvqa", "aokvqa", "vqa", "visual question", "visual question answering",
    "dataset", "benchmark", "direct answer", "answer", "unknown", "not sure", "cannot determine",
}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]+|\d+", str(text).lower())


def is_valid_answer(answer: str, max_words: int = 6) -> bool:
    a = normalize_text(answer)
    if not a or a in INVALID_ANSWERS:
        return False
    toks = normalize_tokens(a)
    if len(toks) == 0 or len(toks) > max_words:
        return False
    # Defensive: reject if it contains only benchmark/task words.
    if all(t in {"a", "okvqa", "vqa", "dataset", "benchmark", "answer", "direct"} for t in toks):
        return False
    return True


def clean_answer(text: str, max_words: int = 4) -> str:
    """Robust answer extraction shared by every method."""
    text = str(text).strip()
    patterns = [
        r"final\s*answer\s*:?\s*(.+)",
        r"answer\s*(?:is)?\s*:?\s*(.+)",
        r"conclusion\s*:?\s*(.+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            ans = m.group(1).strip()
            ans = ans.split("\n")[0].strip()
            ans = re.sub(r"^[\-–—:\s]+", "", ans)
            words = normalize_tokens(ans)
            return " ".join(words[:max_words])

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        words = normalize_tokens(lines[-1])
        return " ".join(words[:max_words])
    words = normalize_tokens(text)
    return " ".join(words[:max_words])


def method_key_to_label(key: str) -> str:
    return {
        "baseline": "Baseline",
        "mcot": "MCoT-v2",
        "vg_crop": "VG-Crop",
        "mcot_vg": "MCoT+VG-v2",
        "router": "MCoT+VG-Router",
    }.get(key, key)
