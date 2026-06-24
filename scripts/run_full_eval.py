#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from mcot_vg.data import category_counts, load_aokvqa_validation
from mcot_vg.grounding import CLIPGradCAMGrounder, GroundingConfig
from mcot_vg.judge import GeminiJudge
from mcot_vg.metrics import accuracy, strict_match
from mcot_vg.models import LlavaConfig, LlavaRunner
from mcot_vg.pipelines import MCOTVGPipelines, Prediction
from mcot_vg.utils import ensure_dir, load_pickle, save_json, save_pickle, set_seed


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1145)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--methods", type=str, default="baseline,mcot,vg_crop,mcot_vg")
    ap.add_argument("--judge", choices=["strict", "gemini"], default="strict")
    ap.add_argument("--gemini_model", type=str, default="gemini-2.5-flash")
    ap.add_argument("--out_dir", type=str, default="outputs/llava_v4")
    ap.add_argument("--no_4bit", action="store_true")
    ap.add_argument("--final_votes", type=int, default=3)
    return ap.parse_args()


def empty_result(method: str):
    return {
        "method": method,
        "answers": [],
        "raw": [],
        "correct": [],
        "correct_strict": [],
        "faith_del": [],
        "faith_ins": [],
        "steps": [],
        "timing": [],
        "categories": [],
        "questions": [],
        "gt_answers": [],
    }


def append_pred(result: dict, sample: dict, pred: Prediction, correct: bool, correct_strict: bool, elapsed: float):
    result["answers"].append(pred.answer)
    result["raw"].append(pred.raw)
    result["correct"].append(bool(correct))
    result["correct_strict"].append(bool(correct_strict))
    result["faith_del"].append(pred.faith_del)
    result["faith_ins"].append(pred.faith_ins)
    result["steps"].append(pred.steps)
    result["timing"].append(float(elapsed))
    result["categories"].append(sample["category"])
    result["questions"].append(sample["question"])
    result["gt_answers"].append(sample["answers"])


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    cache_dir = ensure_dir(out_dir / "cache")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if "router" in methods:
        for dep in ["baseline", "vg_crop", "mcot_vg"]:
            if dep not in methods:
                methods.insert(0, dep)

    samples = load_aokvqa_validation(n=args.n, seed=args.seed)
    save_json({"n": args.n, "seed": args.seed, "category_counts": category_counts(samples), "methods": methods}, out_dir / "run_config.json")

    judge = None
    if args.judge == "gemini":
        judge = GeminiJudge(cache_path=cache_dir / "gemini_judge_cache.jsonl", model_name=args.gemini_model)

    llava = LlavaRunner(LlavaConfig(model_id=args.model_id, load_in_4bit=not args.no_4bit))
    grounder = CLIPGradCAMGrounder(GroundingConfig())
    pipes = MCOTVGPipelines(llava, grounder)

    results = {}
    for method in methods:
        path = ckpt_dir / f"{method}.pkl"
        if path.exists():
            results[method] = load_pickle(path)
        else:
            results[method] = empty_result(method)

    for i, sample in enumerate(tqdm(samples, desc="Evaluating samples")):
        # Run core predictions once per sample; router reuses them.
        per_sample_preds = {}
        for method in methods:
            if method == "router":
                continue
            if len(results[method]["answers"]) > i:
                # Reconstruct minimal prediction for router dependency.
                per_sample_preds[method] = Prediction(
                    answer=results[method]["answers"][i],
                    raw=results[method]["raw"][i],
                    steps=results[method]["steps"][i],
                    faith_del=results[method]["faith_del"][i],
                    faith_ins=results[method]["faith_ins"][i],
                )
                continue
            t0 = time.time()
            if method == "baseline":
                pred = pipes.baseline(sample["image"], sample["question"])
            elif method == "mcot":
                pred = pipes.mcot(sample["image"], sample["question"])
            elif method == "vg_crop":
                pred = pipes.vg_crop(sample["image"], sample["question"])
            elif method == "mcot_vg":
                pred = pipes.mcot_vg(sample["image"], sample["question"], final_votes=args.final_votes)
            else:
                raise ValueError(f"Unknown method: {method}")
            elapsed = time.time() - t0
            correct_strict = strict_match(sample["answers"], pred.answer)
            correct = judge.judge(sample["question"], pred.answer, sample["answers"]) if judge else correct_strict
            append_pred(results[method], sample, pred, correct, correct_strict, elapsed)
            save_pickle(results[method], ckpt_dir / f"{method}.pkl")
            per_sample_preds[method] = pred

        if "router" in methods and len(results["router"]["answers"]) <= i:
            t0 = time.time()
            pred = pipes.router(sample["image"], sample["question"], per_sample_preds["baseline"], per_sample_preds["vg_crop"], per_sample_preds["mcot_vg"])
            elapsed = time.time() - t0
            correct_strict = strict_match(sample["answers"], pred.answer)
            correct = judge.judge(sample["question"], pred.answer, sample["answers"]) if judge else correct_strict
            append_pred(results["router"], sample, pred, correct, correct_strict, elapsed)
            save_pickle(results["router"], ckpt_dir / "router.pkl")

    summary = {
        "primary_judge": args.judge,
        "n": args.n,
        "model_id": args.model_id,
        "accuracy": {m: accuracy(results[m]["correct"]) for m in methods},
        "strict_accuracy": {m: accuracy(results[m]["correct_strict"]) for m in methods},
        "faithfulness": {
            m: {
                "deletion": float(np.nanmean([x for x in results[m]["faith_del"] if x is not None])) if any(x is not None for x in results[m]["faith_del"]) else None,
                "insertion": float(np.nanmean([x for x in results[m]["faith_ins"] if x is not None])) if any(x is not None for x in results[m]["faith_ins"]) else None,
            }
            for m in methods
        },
    }
    save_json(summary, out_dir / "summary_raw.json")
    print("\nSummary:")
    for m in methods:
        print(f"{m:<10} acc={summary['accuracy'][m]:.2f}% strict={summary['strict_accuracy'][m]:.2f}%")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
