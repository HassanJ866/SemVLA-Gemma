"""
Frozen brain inference using vLLM for fast generation + native guided JSON decoding.

vLLM replaces the previous outlines+HF-generate approach. It provides:
  - PagedAttention for efficient KV cache management
  - Native JSON schema-constrained decoding (guided_decoding)
  - Batched generation with continuous batching

Falls back to plain HF generate if vLLM is not installed.

Usage:
    from models.brain.infer import BrainInference
    brain = BrainInference(model_id_or_path="ckpts/brain_phase1/final")
    grounding = brain.ground(image, instruction)
    scene_graph = brain.parse(image, bboxes)
    task = brain.synthesize_task(image, src_name, src_bbox, dst_name, dst_bbox, src_graph)
"""

import json
import logging
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
    Wraps the frozen Gemma 4 E4B brain.
    Uses vLLM for fast inference with native guided JSON decoding.
    Falls back to HF generate if vLLM is unavailable.
    """

    def __init__(
        self,
        model_id_or_path: str,
        device: str | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
    ):
        self.model_id_or_path = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self._backend: str = "none"
        self._load_model()

    def _load_model(self):
        try:
            self._load_vllm()
        except ImportError:
            log.warning("vLLM not installed — falling back to HF generate. "
                        "Install with: pip install vllm")
            self._load_hf()

    def _load_vllm(self):
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor

        log.info(f"Loading brain via vLLM: {self.model_id_or_path}")
        self._llm = LLM(
            model=self.model_id_or_path,
            dtype="bfloat16",
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=2048,
        )
        self._sampling_params_base = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id_or_path, trust_remote_code=True
        )
        # expose embedding table for encode_task_text in chain.py
        self.model = self._llm.llm_engine.model_executor.driver_worker.model_runner.model
        self._backend = "vllm"
        log.info("vLLM backend ready.")

    def _load_hf(self):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        log.info(f"Loading brain via HF: {self.model_id_or_path}")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id_or_path, trust_remote_code=True
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()
        self._backend = "hf"
        log.info("HF backend ready.")

    # ── internal call ─────────────────────────────────────────────────────

    def _call(self, messages: list[dict], image: Image.Image | None,
              schema: dict) -> dict:
        if self._backend == "vllm":
            return self._call_vllm(messages, image, schema)
        return self._call_hf(messages, image, schema)

    def _call_vllm(self, messages: list[dict], image: Image.Image | None,
                   schema: dict) -> dict:
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        guided = GuidedDecodingParams(json=schema)
        sampling = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            guided_decoding=guided,
        )

        inputs: dict[str, Any] = {"prompt": text}
        if image is not None:
            inputs["multi_modal_data"] = {"image": image}

        outputs = self._llm.generate([inputs], sampling_params=sampling)
        result_str = outputs[0].outputs[0].text.strip()
        return json.loads(result_str)

    def _call_hf(self, messages: list[dict], image: Image.Image | None,
                 schema: dict) -> dict:
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
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
            return {"task": "pick up the object and place it at the destination"}
