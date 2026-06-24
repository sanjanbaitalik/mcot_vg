from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import CLIPModel, CLIPProcessor

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_vit_reshape_transform(tensor):
    # CLIP ViT-B/32 has 7x7 patch tokens plus one CLS token.
    tensor = tensor[:, 1:, :]
    result = tensor.reshape(tensor.size(0), 7, 7, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class CLIPWrapperForStep(torch.nn.Module):
    def __init__(self, clip_model, clip_processor, step_text: str, device: str):
        super().__init__()
        self.clip_model = clip_model
        self.text_inputs = clip_processor(
            text=[step_text[:200]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

    def forward(self, pixel_values):
        outputs = self.clip_model(
            pixel_values=pixel_values,
            input_ids=self.text_inputs.input_ids,
            attention_mask=self.text_inputs.attention_mask,
        )
        return outputs.logits_per_image


@dataclass
class GroundingConfig:
    clip_id: str = "openai/clip-vit-base-patch32"
    cam_percentile: float = 80.0
    min_mask_ratio: float = 0.03
    max_mask_ratio: float = 0.80
    crop_pad_frac: float = 0.12
    composite_size: int = 336


class CLIPGradCAMGrounder:
    def __init__(self, cfg: GroundingConfig = GroundingConfig(), device: str | None = None):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = CLIPModel.from_pretrained(cfg.clip_id).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(cfg.clip_id)
        self.clip_model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ])
        self.target_layers = [self.clip_model.vision_model.encoder.layers[-1].layer_norm1]

    def generate(self, image: Image.Image, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (RGB heatmap overlay uint8, grayscale CAM float32 224x224)."""
        image = image.convert("RGB")
        clip_input = self.transform(image).unsqueeze(0).to(self.device)
        image_np = np.array(image.resize((224, 224))).astype(np.float32) / 255.0
        wrapper = CLIPWrapperForStep(self.clip_model, self.clip_processor, text, self.device).to(self.device)
        wrapper.eval()
        cam = GradCAM(
            model=wrapper,
            target_layers=self.target_layers,
            reshape_transform=clip_vit_reshape_transform,
        )
        try:
            grayscale_cam = cam(input_tensor=clip_input)[0]
            grayscale_cam = np.nan_to_num(grayscale_cam).astype(np.float32)
            if grayscale_cam.max() > grayscale_cam.min():
                grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
            vis = show_cam_on_image(image_np, grayscale_cam, use_rgb=True)
        except torch.cuda.OutOfMemoryError:
            grayscale_cam = np.zeros((224, 224), dtype=np.float32)
            vis = (image_np * 255).astype(np.uint8)
        finally:
            del cam, wrapper, clip_input
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return vis, grayscale_cam

    def cam_bbox(self, image: Image.Image, cam: np.ndarray) -> tuple[int, int, int, int]:
        w, h = image.size
        cam_resized = cv2.resize(cam, (w, h))
        if not np.isfinite(cam_resized).all() or cam_resized.max() <= 1e-6:
            return self._center_bbox(w, h)
        thr = np.percentile(cam_resized, self.cfg.cam_percentile)
        mask = cam_resized >= thr
        ratio = mask.mean()
        if ratio < self.cfg.min_mask_ratio or ratio > self.cfg.max_mask_ratio:
            return self._center_bbox(w, h)
        ys, xs = np.where(mask)
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad = int(max(x2 - x1, y2 - y1) * self.cfg.crop_pad_frac) + 2
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            return self._center_bbox(w, h)
        return x1, y1, x2, y2

    @staticmethod
    def _center_bbox(w: int, h: int) -> tuple[int, int, int, int]:
        side = int(min(w, h) * 0.65)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, cx + side // 2)
        y2 = min(h, cy + side // 2)
        return x1, y1, x2, y2

    def crop_from_cam(self, image: Image.Image, cam: np.ndarray) -> Image.Image:
        bbox = self.cam_bbox(image, cam)
        crop = image.crop(bbox).convert("RGB")
        return ImageOps.expand(crop, border=4, fill="white")

    def composite_full_crop(self, image: Image.Image, crop: Image.Image) -> Image.Image:
        """Single image for LLaVA: full image left, zoomed crop right."""
        size = self.cfg.composite_size
        full = ImageOps.contain(image.convert("RGB"), (size, size), method=Image.BICUBIC)
        crop = ImageOps.contain(crop.convert("RGB"), (size, size), method=Image.BICUBIC)
        canvas = Image.new("RGB", (size * 2 + 12, size), "white")
        canvas.paste(full, (0, (size - full.height) // 2))
        canvas.paste(crop, (size + 12, (size - crop.height) // 2))
        return canvas
