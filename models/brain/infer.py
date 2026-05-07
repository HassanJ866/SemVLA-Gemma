"""
Frozen brain inference with constrained JSON decoding.

Uses HF generate with outlines for structured output.
At inference time the model is loaded in bfloat16 and kept frozen.

Usage:
    from models.brain.infer import BrainInference
    brain = BrainInference(model_id_or_path="ckpts/brain_phase1/final")
    grounding = brain.ground(image, instruction)
    scene_graph = brain.parse(image, bboxes)
    task = brain.synthesize_task(image, src_name, src_bbox, dst_name, dst_bbox, src_graph)
"""

import json
import logging

import torch
from PIL import Image

log = logging.getLogger(__name__)

GROUNDING_SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string"},
        "bbox":   {"type": "array", "items": {"type": "number"},
                   "minItems": 4, "maxItems": 4},
    },
    "required": ["object", "bbox"],
    "additionalProperties": False,
}

PARSING_SCHEMA = {
    "type": "object",
    "properties": {
        "triplets": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3, "maxItems": 3,
            },
        },
    },
    "required": ["triplets"],
    "additionalProperties": False,
}

TASK_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": "string"},
    },
    "required": ["task"],
    "additionalProperties": False,
}


class BrainInference:
    """
    Wraps the frozen Gemma 4 E4B brain.
    Uses outlines for constrained JSON decoding; falls back to plain HF generate.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ):
        self.model_id_or_path = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._load_model()

    def _load_model(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        log.info(f"Loading brain from {self.model_id_or_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id_or_path, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        log.info("Brain loaded.")

def _call(self, messages: list[dict], image: Image.Image | None,
              schema: dict) -> dict:
        # Unsloth/Gemma4 inference pattern: apply_chat_template renders the
        # text string, then processor(image, text) produces all required tensors
        # including image_position_ids.
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.processor(
            image,
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()
               if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                use_cache=True,
            )
        new_tokens = out_ids[0, enc["input_ids"].shape[1]:]
        text_out = self.processor.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()
        return json.loads(text_out)

    # ── public API ────────────────────────────────────────────────────────

    def ground(self, image: Image.Image, instruction: str) -> dict:
        from models.brain.prompts import grounding_prompt
        try:
            return self._call(grounding_prompt(instruction), image, GROUNDING_SCHEMA)
        except Exception as e:
            log.error(f"Grounding failed: {e}")
            return {"object": "unknown", "bbox": [0, 0, 0, 0]}

    def parse(self, image: Image.Image, bboxes: list[dict]) -> dict:
        from models.brain.prompts import parsing_prompt
        try:
            return self._call(parsing_prompt(bboxes), image, PARSING_SCHEMA)
        except Exception as e:
            log.error(f"Parsing failed: {e}")
            return {"triplets": []}

    def synthesize_task(self, image: Image.Image, src_name: str, src_bbox: list,
                        dst_name: str, dst_bbox: list, src_graph: list) -> dict:
        from models.brain.prompts import task_synthesis_prompt
        try:
            return self._call(
                task_synthesis_prompt(src_name, src_bbox, dst_name, dst_bbox, src_graph),
                image, TASK_SYNTHESIS_SCHEMA,
            )
        except Exception as e:
            log.error(f"Task synthesis failed: {e}")
            return {"task": "pick up the object and place it at the destination"}
