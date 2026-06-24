#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mcot_vg.claims import generate_claims
from mcot_vg.metrics import accuracy, bootstrap_delta_ci, mcnemar_test, per_category_accuracy
from mcot_vg.utils import ensure_dir, load_json, load_pickle, method_key_to_label, save_json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    report_dir = ensure_dir(out_dir / "reports")

    methods = [p.stem for p in sorted(ckpt_dir.glob("*.pkl"))]
    results = {m: load_pickle(ckpt_dir / f"{m}.pkl") for m in methods}
    methods = [m for m in ["baseline", "mcot", "vg_crop", "mcot_vg", "router"] if m in results]

    summary_raw = load_json(out_dir / "summary_raw.json") if (out_dir / "summary_raw.json").exists() else {}
    summary = {
        "primary_judge": summary_raw.get("primary_judge", "strict"),
        "n": len(next(iter(results.values()))["correct"]),
        "model_id": summary_raw.get("model_id", "unknown"),
        "accuracy": {m: accuracy(results[m]["correct"]) for m in methods},
        "strict_accuracy": {m: accuracy(results[m]["correct_strict"]) for m in methods},
        "faithfulness": {},
        "mcnemar": {},
        "bootstrap_delta_ci": {},
        "per_category": {},
        "timing": {},
    }

    for m in methods:
        fd = [x for x in results[m]["faith_del"] if x is not None]
        fi = [x for x in results[m]["faith_ins"] if x is not None]
        summary["faithfulness"][m] = {
            "deletion": float(np.mean(fd)) if fd else None,
            "insertion": float(np.mean(fi)) if fi else None,
        }
        summary["per_category"][m] = per_category_accuracy(results[m]["correct"], results[m]["categories"])
        summary["timing"][m] = float(np.mean(results[m]["timing"])) if results[m]["timing"] else None

    comparisons = [
        ("baseline", "mcot_vg", "baseline_vs_mcot_vg"),
        ("vg_crop", "mcot_vg", "vg_crop_vs_mcot_vg"),
        ("mcot", "mcot_vg", "mcot_vs_mcot_vg"),
        ("baseline", "vg_crop", "baseline_vs_vg_crop"),
        ("mcot_vg", "router", "mcot_vg_vs_router"),
    ]
    for a, b, name in comparisons:
        if a in results and b in results:
            summary["mcnemar"][name] = mcnemar_test(results[a]["correct"], results[b]["correct"])
            summary["bootstrap_delta_ci"][name] = bootstrap_delta_ci(results[a]["correct"], results[b]["correct"], seed=42)

    save_json(summary, out_dir / "summary_report.json")

    # Markdown table
    lines = []
    lines.append(f"# MCoT+VG Evaluation Report")
    lines.append("")
    lines.append(f"Model: `{summary['model_id']}`  ")
    lines.append(f"N: `{summary['n']}`  ")
    lines.append(f"Primary judge: `{summary['primary_judge']}`")
    lines.append("")
    lines.append("## Main metrics")
    lines.append("")
    lines.append("| Method | Primary Acc. | Strict Acc. | Faith Deletion ↑ | Faith Insertion ↑ | Mean time/sample |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for m in methods:
        f = summary["faithfulness"][m]
        fd = "—" if f["deletion"] is None else f"{f['deletion']:.5f}"
        fi = "—" if f["insertion"] is None else f"{f['insertion']:.5f}"
        t = "—" if summary["timing"][m] is None else f"{summary['timing'][m]:.2f}s"
        lines.append(f"| {method_key_to_label(m)} | {summary['accuracy'][m]:.2f}% | {summary['strict_accuracy'][m]:.2f}% | {fd} | {fi} | {t} |")

    lines.append("")
    lines.append("## Statistical tests")
    lines.append("")
    lines.append("| Comparison | Δ accuracy (B - A) | 95% bootstrap CI | McNemar p |")
    lines.append("|---|---:|---:|---:|")
    for name, ci in summary["bootstrap_delta_ci"].items():
        p = summary["mcnemar"][name]["p"]
        lines.append(f"| {name.replace('_', ' ')} | {ci['delta']:.2f} | [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}] | {p:.4f} |")

    lines.append("")
    lines.append("## Per-category primary accuracy")
    lines.append("")
    cats = sorted(set().union(*[set(summary["per_category"][m].keys()) for m in methods]))
    lines.append("| Category | " + " | ".join(method_key_to_label(m) for m in methods) + " |")
    lines.append("|---" + "|---:" * len(methods) + "|")
    for c in cats:
        vals = [summary["per_category"][m].get(c) for m in methods]
        lines.append("| " + c + " | " + " | ".join("—" if v is None else f"{v:.2f}%" for v in vals) + " |")

    report_md = "\n".join(lines) + "\n"
    (report_dir / "metrics_report.md").write_text(report_md, encoding="utf-8")

    claims = generate_claims(summary)
    (report_dir / "paper_claims.md").write_text(claims, encoding="utf-8")
    print(report_md)
    print("\nGenerated:")
    print(report_dir / "metrics_report.md")
    print(report_dir / "paper_claims.md")


if __name__ == "__main__":
    main()
