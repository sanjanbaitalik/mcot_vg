from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from .utils import normalize_tokens


def strict_match(gt_answers: Sequence[str], prediction: str) -> bool:
    """Conservative direct-answer matching for debugging; Gemini is primary if enabled."""
    pred_tokens = normalize_tokens(prediction)
    pred_str = " ".join(pred_tokens)
    if not pred_str:
        return False
    for gt in gt_answers:
        gt_tokens = normalize_tokens(gt)
        gt_str = " ".join(gt_tokens)
        if not gt_str:
            continue
        if gt_str == pred_str:
            return True
        if gt_str in pred_str or pred_str in gt_str:
            return True
        if len(gt_tokens) == 1 and len(gt_tokens[0]) >= 5:
            stem = gt_tokens[0][:5]
            if any(t[:5] == stem for t in pred_tokens):
                return True
        if len(gt_tokens) > 1:
            overlap = sum(1 for t in gt_tokens if t in pred_tokens)
            if overlap / max(1, len(gt_tokens)) >= 0.75:
                return True
    return False


def accuracy(res: Sequence[bool]) -> float:
    return 100.0 * float(np.mean([bool(x) for x in res])) if res else 0.0


def mcnemar_test(res_a: Sequence[bool], res_b: Sequence[bool]) -> Dict[str, float]:
    n01 = sum(1 for a, b in zip(res_a, res_b) if (not a) and b)
    n10 = sum(1 for a, b in zip(res_a, res_b) if a and (not b))
    denom = n01 + n10
    stat = ((abs(n01 - n10) - 1) ** 2 / denom) if denom > 0 else 0.0
    p = float(1 - chi2.cdf(stat, df=1))
    return {"n01": n01, "n10": n10, "chi2": float(stat), "p": p}


def bootstrap_delta_ci(res_a: Sequence[bool], res_b: Sequence[bool], n_boot: int = 10000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(res_a, dtype=float)
    b = np.asarray(res_b, dtype=float)
    n = len(a)
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (b[idx].mean(axis=1) - a[idx].mean(axis=1)) * 100.0
    return {
        "delta": float((b.mean() - a.mean()) * 100.0),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
    }


def per_category_accuracy(results: Sequence[bool], categories: Sequence[str]) -> Dict[str, float]:
    correct = defaultdict(int)
    total = defaultdict(int)
    for r, c in zip(results, categories):
        total[c] += 1
        correct[c] += int(bool(r))
    return {c: 100.0 * correct[c] / total[c] for c in sorted(total)}


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None
