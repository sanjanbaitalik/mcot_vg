from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration


@dataclass
class LlavaConfig:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    load_in_4bit: bool = True
    torch_dtype: str = "float16"


class LlavaRunner:
    """Thin wrapper around HuggingFace LLaVA-style models.

    This package defaults to LLaVA-1.5-7B because the user's current run uses it.
    If using LLaVA-1.6/LLaVA-NeXT, adapt this wrapper to LlavaNextForConditionalGeneration.
    """

    def __init__(self, cfg: LlavaConfig):
        self.cfg = cfg
        dtype = torch.float16 if cfg.torch_dtype == "float16" else torch.bfloat16
        quant = None
        if cfg.load_in_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            cfg.model_id,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=dtype,
        )
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def generate(
        self,
        image: Image.Image,
        prompt_text: str,
        max_new_tokens: int = 96,
        temperature: Optional[float] = None,
        do_sample: bool = False,
        top_p: float = 0.9,
    ) -> str:
        conversation = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": prompt_text}],
        }]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = temperature if temperature is not None else 0.3
            kwargs["top_p"] = top_p
        output_ids = self.model.generate(**inputs, **kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[-1]:]
        text = self.processor.decode(generated[0], skip_special_tokens=True).strip()
        del inputs, output_ids, generated
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text
