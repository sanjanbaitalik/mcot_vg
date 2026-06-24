from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .faithfulness import FaithfulnessScorer
from .grounding import CLIPGradCAMGrounder
from .models import LlavaRunner
from .prompts import (
    baseline_prompt,
    fallback_final_prompt,
    final_answer_prompt,
    mcot_prompt_v2,
    router_prompt,
    step1_visual_evidence_prompt,
    step2_grounded_evidence_prompt,
    vg_crop_prompt,
)
from .utils import clean_answer, is_valid_answer


@dataclass
class Prediction:
    answer: str
    raw: str
    steps: list[str]
    faith_del: Optional[float] = None
    faith_ins: Optional[float] = None
    meta: Optional[Dict] = None


class MCOTVGPipelines:
    def __init__(self, llava: LlavaRunner, grounder: CLIPGradCAMGrounder):
        self.llava = llava
        self.grounder = grounder
        self.faith = FaithfulnessScorer(grounder)

    def baseline(self, image, question: str) -> Prediction:
        raw = self.llava.generate(image, baseline_prompt(question), max_new_tokens=48)
        return Prediction(answer=clean_answer(raw), raw=raw, steps=[])

    def mcot(self, image, question: str) -> Prediction:
        raw = self.llava.generate(image, mcot_prompt_v2(question), max_new_tokens=180)
        return Prediction(answer=clean_answer(raw), raw=raw, steps=[raw])

    def vg_crop(self, image, question: str) -> Prediction:
        vis, cam = self.grounder.generate(image, question)
        crop = self.grounder.crop_from_cam(image, cam)
        composite = self.grounder.composite_full_crop(image, crop)
        raw = self.llava.generate(composite, vg_crop_prompt(question), max_new_tokens=48)
        f_del, f_ins = self.faith.score(image, question, cam)
        return Prediction(answer=clean_answer(raw), raw=raw, steps=[question], faith_del=f_del, faith_ins=f_ins, meta={"has_crop": True})

    def _choose_final_answer(self, raw_votes: list[str]) -> tuple[str, list[str]]:
        answers = [clean_answer(x) for x in raw_votes]
        valid = [a for a in answers if is_valid_answer(a)]
        if valid:
            return Counter(valid).most_common(1)[0][0], answers
        return answers[0] if answers else "", answers

    def mcot_vg(self, image, question: str, final_votes: int = 3) -> Prediction:
        steps = []

        # Step 1: text evidence from the original image.
        step1 = self.llava.generate(image, step1_visual_evidence_prompt(question), max_new_tokens=96)
        steps.append(step1)

        # Step 1 grounding and crop.
        cam_query1 = f"{question}. {step1[:160]}"
        _, cam1 = self.grounder.generate(image, cam_query1)
        crop1 = self.grounder.crop_from_cam(image, cam1)
        composite1 = self.grounder.composite_full_crop(image, crop1)

        # Step 2: grounded evidence from original + crop.
        step2 = self.llava.generate(composite1, step2_grounded_evidence_prompt(question, step1), max_new_tokens=120)
        steps.append(step2)

        # Step 2 grounding and crop.
        cam_query2 = f"{question}. {step2[:180]}"
        _, cam2 = self.grounder.generate(image, cam_query2)
        crop2 = self.grounder.crop_from_cam(image, cam2)
        composite2 = self.grounder.composite_full_crop(image, crop2)

        # Step 3: final answer. Keep the prompt dataset-name-free to prevent answer leakage.
        prompt = final_answer_prompt(question, step1, step2)
        raw_votes = [self.llava.generate(composite2, prompt, max_new_tokens=48)]
        for _ in range(max(0, final_votes - 1)):
            raw_votes.append(self.llava.generate(composite2, prompt, max_new_tokens=48, do_sample=True, temperature=0.25))

        final, cleaned_votes = self._choose_final_answer(raw_votes)
        raw0 = raw_votes[0]

        # Defensive fallback: if the model outputs a dataset/task label, ask a simpler final-only question.
        if not is_valid_answer(final):
            fallback_raw = self.llava.generate(composite2, fallback_final_prompt(question), max_new_tokens=32)
            fallback_ans = clean_answer(fallback_raw)
            if not is_valid_answer(fallback_ans):
                fallback_raw = self.llava.generate(image, baseline_prompt(question), max_new_tokens=32)
                fallback_ans = clean_answer(fallback_raw)
            final = fallback_ans
            raw0 = fallback_raw
            raw_votes.append(fallback_raw)
            cleaned_votes.append(fallback_ans)

        # Step-3 CAM: final answer query, to make all three stages grounded.
        _, cam3 = self.grounder.generate(image, f"{question}. {final}")
        f_del_1, f_ins_1 = self.faith.score(image, cam_query1, cam1)
        f_del_2, f_ins_2 = self.faith.score(image, cam_query2, cam2)
        f_del_3, f_ins_3 = self.faith.score(image, f"{question}. {final}", cam3)
        f_del = float(np.mean([f_del_1, f_del_2, f_del_3]))
        f_ins = float(np.mean([f_ins_1, f_ins_2, f_ins_3]))
        steps.append(raw0)
        return Prediction(
            answer=final,
            raw=raw0,
            steps=steps,
            faith_del=f_del,
            faith_ins=f_ins,
            meta={"raw_votes": raw_votes, "cleaned_votes": cleaned_votes},
        )

    def router(self, image, question: str, baseline_pred: Prediction, vg_pred: Prediction, mcot_vg_pred: Prediction) -> Prediction:
        """Optional accuracy-oriented selector. Report separately from the core method."""
        raw_candidates = {
            "Baseline": baseline_pred.answer,
            "VG-Crop": vg_pred.answer,
            "MCoT+VG": mcot_vg_pred.answer,
        }
        candidates = {k: v for k, v in raw_candidates.items() if is_valid_answer(v)}
        if not candidates:
            return Prediction(answer=baseline_pred.answer, raw=baseline_pred.answer, steps=[], meta={"router": "fallback", "candidates": raw_candidates})

        # Exact majority first, ignoring invalid candidates.
        counts = Counter(candidates.values())
        ans, cnt = counts.most_common(1)[0]
        if cnt >= 2:
            return Prediction(answer=ans, raw=ans, steps=[], meta={"router": "majority", "candidates": raw_candidates})

        # If all valid candidates disagree, use the image-aware selector.
        raw = self.llava.generate(image, router_prompt(question, candidates), max_new_tokens=48)
        ans = clean_answer(raw)
        if not is_valid_answer(ans):
            ans = baseline_pred.answer
            raw = baseline_pred.raw
        return Prediction(answer=ans, raw=raw, steps=[], meta={"router": "llava_selector", "candidates": raw_candidates})
