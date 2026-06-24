#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mcot_vg.utils import ensure_dir, load_pickle


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--method", default="mcot_vg")
    ap.add_argument("--n_items", type=int, default=100)
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ckpt = out_dir / "checkpoints" / f"{args.method}.pkl"
    data = load_pickle(ckpt)
    rows = []
    for i in range(min(args.n_items, len(data["answers"]))):
        steps = data["steps"][i]
        for j, step in enumerate(steps, start=1):
            rows.append({
                "sample_id": i,
                "step_id": j,
                "question": data["questions"][i],
                "ground_truth_answers": " | ".join(data["gt_answers"][i]),
                "model_answer": data["answers"][i],
                "step_text": step,
                "HIS_annotator_1_1to5": "",
                "HIS_annotator_2_1to5": "",
                "HIS_annotator_3_1to5": "",
                "HIS_annotator_4_1to5": "",
                "HIS_annotator_5_1to5": "",
                "SC_annotator_1_0or1": "",
                "SC_annotator_2_0or1": "",
                "SC_annotator_3_0or1": "",
                "SC_annotator_4_0or1": "",
                "SC_annotator_5_0or1": "",
            })
    path = ensure_dir(out_dir / "annotations") / f"{args.method}_annotation_template.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
