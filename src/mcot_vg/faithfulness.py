from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass
class FaithfulnessConfig:
    tau: float = 0.50
    min_mask_ratio: float = 0.03
    max_mask_ratio: float = 0.90


class FaithfulnessScorer:
    """CLIP cosine deletion/insertion faithfulness.

    Deletion: max(0, sim(original, text) - sim(image with highlighted region removed, text)).
    Insertion: sim(image containing only highlighted region, text).
    """

    def __init__(self, grounder, cfg: FaithfulnessConfig = FaithfulnessConfig()):
        self.grounder = grounder
        self.cfg = cfg
        self.device = grounder.device
        self.clip_model = grounder.clip_model
        self.clip_processor = grounder.clip_processor

    @torch.no_grad()
    def cosine_clip_similarity(self, image: Image.Image, text: str) -> float:
        image = image.convert("RGB").resize((224, 224))
        text = str(text)[:200]
        img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        txt_inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        vision_out = self.clip_model.vision_model(pixel_values=img_inputs["pixel_values"])
        img_feat = self.clip_model.visual_projection(vision_out.pooler_output)
        text_out = self.clip_model.text_model(input_ids=txt_inputs["input_ids"], attention_mask=txt_inputs["attention_mask"])
        txt_feat = self.clip_model.text_projection(text_out.pooler_output)
        img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-8)
        txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (img_feat * txt_feat).sum().item()
        return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))

    def score(self, image: Image.Image, text: str, cam: np.ndarray) -> tuple[float, float]:
        img_np = np.array(image.convert("RGB").resize((224, 224))).astype(np.float32) / 255.0
        cam_resized = cv2.resize(cam, (224, 224))
        mask = cam_resized >= self.cfg.tau
        ratio = float(mask.mean())
        if ratio < self.cfg.min_mask_ratio or ratio > self.cfg.max_mask_ratio:
            return 0.0, 0.0

        deleted = img_np.copy()
        deleted[mask] = 0.0
        deleted_pil = Image.fromarray((deleted * 255).astype(np.uint8))

        inserted = np.zeros_like(img_np)
        inserted[mask] = img_np[mask]
        inserted_pil = Image.fromarray((inserted * 255).astype(np.uint8))

        sim_orig = self.cosine_clip_similarity(image, text)
        sim_deleted = self.cosine_clip_similarity(deleted_pil, text)
        sim_inserted = self.cosine_clip_similarity(inserted_pil, text)
        return float(max(0.0, sim_orig - sim_deleted)), float(sim_inserted)
