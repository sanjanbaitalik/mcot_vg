from __future__ import annotations

import ast
import random
from collections import defaultdict
from typing import Dict, List

from datasets import load_dataset
from tqdm.auto import tqdm

CATEGORY_KEYWORDS = {
    "animals": ["dog", "cat", "bird", "horse", "cow", "elephant", "bear", "zebra", "giraffe", "sheep", "goat", "duck", "rabbit", "fish", "animal"],
    "sports": ["sport", "ball", "player", "bat", "racket", "tennis", "baseball", "football", "soccer", "basketball", "skateboard", "ski", "snowboard"],
    "household": ["kitchen", "table", "chair", "sofa", "bed", "room", "plate", "cup", "bottle", "appliance", "sink", "toilet", "laptop"],
    "scenes": ["beach", "mountain", "street", "road", "park", "forest", "field", "sky", "snow", "water", "city", "building"],
    "vehicles": ["car", "bus", "truck", "train", "bike", "bicycle", "motorcycle", "airplane", "boat", "vehicle"],
}


def infer_category(question: str) -> str:
    q = question.lower()
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(k in q for k in kws):
            return cat
    return "other"


def _parse_answers(ans):
    if isinstance(ans, str):
        try:
            ans = ast.literal_eval(ans)
        except Exception:
            ans = [ans]
    if not isinstance(ans, list):
        ans = [str(ans)]
    return [str(a) for a in ans]


def load_aokvqa_validation(n: int = 1145, seed: int = 42, streaming: bool = True) -> List[Dict]:
    """Load a fixed shuffled A-OKVQA validation subset.

    All methods must use this same list object/order to ensure N parity.
    """
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation", streaming=streaming)
    samples = []
    for x in tqdm(ds, total=n, desc="Loading A-OKVQA validation"):
        img = x["image"].convert("RGB")
        q = str(x["question"])
        ans = _parse_answers(x["direct_answers"])
        samples.append({
            "image": img,
            "question": q,
            "answers": ans,
            "category": infer_category(q),
        })
        if len(samples) >= n:
            break
    if len(samples) != n:
        raise RuntimeError(f"Expected {n} samples, got {len(samples)}")
    random.seed(seed)
    random.shuffle(samples)
    return samples


def category_counts(samples: List[Dict]) -> Dict[str, int]:
    d = defaultdict(int)
    for s in samples:
        d[s.get("category", "other")] += 1
    return dict(d)
