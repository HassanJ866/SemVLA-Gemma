"""
Frozen brain inference with constrained JSON decoding (outlines).

Usage:
    from models.brain.infer import BrainInference
    brain = BrainInference(model_id_or_path="ckpts/brain_phase1/final")
    grounding = brain.ground(image, instruction)
    scene_graph = brain.parse(image, bboxes)
    task = brain.synthesize_task(image, src_name, src_bbox, dst_name, dst_bbox, src_graph)
"""

import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image

log = logging.getLogger(__name__)

# JSON schemas for constrained decoding
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
    Wraps the frozen Gemma 4 brain for the three inference tasks.
    Uses `outlines` for constrained JSON decoding.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        use_constrained_decoding: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_constrained = use_constrained_decoding
        self._load_model(model_id_or_path)

    def _load_model(self, path: str):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel

        log.info(f"Loading brain from {path}")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)

        base_model = AutoModelForImageTextToText.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        base_model.eval()
        self.model = base_model

        # build constrained generators lazily
        self._generators: dict[str, Any] = {}

    def _get_generator(self, schema: dict):
        key = json.dumps(schema, sort_keys=True)
        if key not in self._generators:
            try:
                import outlines
                import outlines.models as omodels
                wrapped = omodels.Transformers(self.model, self.processor.tokenizer)
                gen = outlines.generate.json(wrapped, schema)
                self._generators[key] = gen
            except Exception as e:
                log.warning(f"Could not build outlines generator ({e}); "
                            "falling back to unconstrained greedy decoding")
                self._generators[key] = None
        return self._generators[key]

    def _call(self, messages: list[dict], image: Image.Image | None,
              schema: dict) -> dict:
        from models.brain.prompts import format_training_sample
        gen = self._get_generator(schema) if self.use_constrained else None

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if gen is not None:
            # outlines handles image via transformers processor internally
            result_str = gen(text, images=[image] if image is not None else None)
            if isinstance(result_str, dict):
                return result_str
            return json.loads(result_str)

        # fallback: standard HF generate
        enc = self.processor(
            text=[text],
            images=[image] if image is not None else None,
            return_tensors="pt",
            padding=True,
        )
        enc = {k: v.to(self.device) for k, v in enc.items() if v is not None}

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
            )
        new_tokens = out_ids[0, enc["input_ids"].shape[1]:]
        text_out = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        return json.loads(text_out)

    # ── public API ────────────────────────────────────────────────────────

    def ground(self, image: Image.Image, instruction: str) -> dict:
        """Returns {"object": str, "bbox": [x1,y1,x2,y2]}"""
        from models.brain.prompts import grounding_prompt
        messages = grounding_prompt(instruction)
        try:
            return self._call(messages, image, GROUNDING_SCHEMA)
        except Exception as e:
            log.error(f"Grounding failed: {e}")
            return {"object": "unknown", "bbox": [0, 0, 0, 0]}

    def parse(self, image: Image.Image, bboxes: list[dict]) -> dict:
        """Returns {"triplets": [[s, r, o], ...]}"""
        from models.brain.prompts import parsing_prompt
        messages = parsing_prompt(bboxes)
        try:
            return self._call(messages, image, PARSING_SCHEMA)
        except Exception as e:
            log.error(f"Parsing failed: {e}")
            return {"triplets": []}

    def synthesize_task(self, image: Image.Image, src_name: str, src_bbox: list,
                        dst_name: str, dst_bbox: list, src_graph: list) -> dict:
        """Returns {"task": <natural language instruction string>}"""
        from models.brain.prompts import task_synthesis_prompt
        messages = task_synthesis_prompt(src_name, src_bbox, dst_name, dst_bbox, src_graph)
        try:
            return self._call(messages, image, TASK_SYNTHESIS_SCHEMA)
        except Exception as e:
            log.error(f"Task synthesis failed: {e}; returning fallback")
            return {"task": f"pick up the object and place it at the destination"}
