"""
5-step inference chain (middleware).

Implements the full per-control-step decision pipeline:
  Step 1  GROUNDING      — brain LLM call: image + instruction → target bbox
  Step 2  PARSING        — brain LLM call: image + bboxes → scene graph
  Step 3  TASK SYNTHESIS — brain LLM call: image + src/dst bboxes + local graph → task string
  Step 4  MIDDLEWARE     — encode task string → task_emb; encode graph → graph tensor
  Step 5  ADAPTER        — flow matching inference → denormalised action chunk

Usage:
    from models.middleware.chain import InferenceChain
    chain = InferenceChain(brain_ckpt="ckpts/brain_phase1/final",
                           adapter_ckpt="ckpts/franka_7dof",
                           action_stats="ckpts/franka_7dof/action_stats.json")
    action_chunk = chain.step(image, instruction, proprio)
    raw_action   = action_chunk[0]  # pop first from chunk
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from models.middleware.graph_encoder import encode_graph_tensor, GRAPH_FEAT_DIM
from models.middleware.normalize import ActionNormalizer

log = logging.getLogger(__name__)


TASK_EMBED_DIM = 384  # mean-pooled Gemma token embedding dimension
SAFE_STOP_ACTION = np.zeros(7, dtype=np.float32)  # zero delta = stay in place


def encode_task_text(text: str, tokenizer, embed_table: torch.Tensor,
                     device: str = "cpu") -> torch.Tensor:
    """
    Encode a task string to a 384-dim float vector via mean-pooled Gemma token embeddings.
    Uses the frozen brain's own embedding table — no extra model required.

    Returns tensor of shape [1, TASK_EMBED_DIM].
    """
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = tokens["input_ids"][0]  # [L]
    with torch.no_grad():
        embs = embed_table(input_ids.to(embed_table.weight.device))  # [L, vocab_emb_dim]
        pooled = embs.mean(dim=0)  # [vocab_emb_dim]
        # project to TASK_EMBED_DIM if vocab embedding dim differs
        if pooled.shape[0] != TASK_EMBED_DIM:
            pooled = pooled[:TASK_EMBED_DIM] if pooled.shape[0] > TASK_EMBED_DIM else \
                     torch.nn.functional.pad(pooled, (0, TASK_EMBED_DIM - pooled.shape[0]))
    return pooled.to(device).unsqueeze(0)  # [1, TASK_EMBED_DIM]


class InferenceChain:
    """
    Stateless, single-step inference chain.

    Parameters
    ----------
    brain_ckpt      Path to frozen brain checkpoint.
    adapter_ckpt    Path to adapter checkpoint directory.
    action_stats    Path to action_stats.json for denormalisation.
    n_flow_steps    Number of Euler denoising steps (default: 10).
    device          Torch device string.
    cache_graph     If True, reuse the scene graph across N calls
                    (set cache_every > 1 for latency optimisation).
    cache_every     How many control steps to reuse a cached graph.
    """

    def __init__(
        self,
        brain_ckpt: str,
        adapter_ckpt: str,
        action_stats: str,
        n_flow_steps: int = 10,
        device: str | None = None,
        cache_graph: bool = False,
        cache_every: int = 5,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_flow_steps = n_flow_steps
        self.cache_graph = cache_graph
        self.cache_every = cache_every
        self._cached_graph: dict | None = None
        self._cache_counter: int = 0

        from models.brain.infer import BrainInference
        self.brain = BrainInference(brain_ckpt, device=self.device)

        self.normalizer = ActionNormalizer(action_stats)

        from models.adapter.architecture import SemanticActionAdapter
        ckpt_path = Path(adapter_ckpt)
        cfg_path = ckpt_path / "adapter_config.json"
        import json
        with open(cfg_path) as f:
            adapter_cfg = json.load(f)

        self.adapter = SemanticActionAdapter(**adapter_cfg).to(self.device)
        state_dict = torch.load(ckpt_path / "adapter.pt",
                                map_location=self.device, weights_only=True)
        self.adapter.load_state_dict(state_dict)
        self.adapter.eval()

        self.action_dim     = adapter_cfg["action_dim"]
        self.chunk_size     = adapter_cfg.get("chunk_size", 16)
        self.graph_feat_dim = adapter_cfg.get("graph_feat_dim", GRAPH_FEAT_DIM)
        self.task_embed_dim = adapter_cfg.get("task_embed_dim", TASK_EMBED_DIM)

        # cache brain embedding table + tokenizer for task text encoding
        self._embed_table = self.brain.model.get_input_embeddings()
        self._tokenizer   = self.brain.processor.tokenizer

    # ── public ────────────────────────────────────────────────────────────

    def step(
        self,
        image: Image.Image,
        instruction: str,
        proprio: list[float] | np.ndarray,
    ) -> np.ndarray:
        """
        Run one full control step. Returns a denormalised action chunk
        of shape [chunk_size, action_dim].
        Falls back to SAFE_STOP_ACTION on any unrecoverable failure.
        """
        try:
            return self._step_impl(image, instruction, proprio)
        except Exception as e:
            log.error(f"InferenceChain.step failed: {e}; returning safe-stop chunk")
            chunk = np.tile(SAFE_STOP_ACTION[:self.action_dim],
                            (self.chunk_size, 1))
            return chunk

    def _step_impl(
        self,
        image: Image.Image,
        instruction: str,
        proprio: list[float] | np.ndarray,
    ) -> np.ndarray:
        # Step 1 — GROUNDING
        grounding = self.brain.ground(image, instruction)
        target_bbox = grounding.get("bbox", [0, 0, 0, 0])

        # Step 2 — PARSING
        if self.cache_graph and self._cached_graph is not None:
            scene_graph = self._cached_graph
            self._cache_counter += 1
            if self._cache_counter >= self.cache_every:
                self._cached_graph = None
                self._cache_counter = 0
        else:
            bboxes = [{"name": grounding.get("object", "target"), "bbox": target_bbox}]
            scene_graph = self.brain.parse(image, bboxes)
            if self.cache_graph:
                self._cached_graph = scene_graph

        proprio_list = proprio.tolist() if isinstance(proprio, np.ndarray) else list(proprio)

        # Step 3 — TASK SYNTHESIS
        # Use the grounding output to identify source object + build its local graph.
        src_name  = grounding.get("object", "")
        src_bbox  = grounding.get("bbox", [0, 0, 0, 0])
        triplets  = scene_graph.get("triplets", [])
        src_graph = [t for t in triplets if len(t) == 3 and t[0] == src_name]

        # Destination: pick the first object in the scene graph that isn't the source.
        # At runtime the instruction disambiguates; this is a best-effort heuristic.
        dst_name = src_name
        dst_bbox = src_bbox
        bboxes_in_graph = list({t[0] for t in triplets if len(t) == 3} |
                               {t[2] for t in triplets if len(t) == 3})
        for candidate in bboxes_in_graph:
            if candidate != src_name:
                dst_name = candidate
                dst_bbox = [0, 0, 0, 0]  # no bbox for dst at runtime; brain uses visual context
                break

        task_result = self.brain.synthesize_task(
            image, src_name, src_bbox, dst_name, dst_bbox, src_graph
        )
        self._validate_schema(task_result, ["task"])
        task_text = task_result["task"]

        # Step 4 — MIDDLEWARE: encode task text + graph
        task_emb_tensor = encode_task_text(
            task_text, self._tokenizer, self._embed_table, device=self.device
        )  # [1, task_embed_dim]

        graph_tensor = encode_graph_tensor(
            triplets,
            dim=self.graph_feat_dim,
            device=self.device,
        )  # [1, 1, G]

        state_tensor = torch.tensor(
            proprio_list, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, state_dim]

        # Step 5 — ADAPTER: flow matching inference
        from models.adapter.flow_matching import flow_matching_inference
        with torch.no_grad():
            normed_chunk = flow_matching_inference(
                self.adapter, task_emb_tensor, graph_tensor, state_tensor,
                self.chunk_size, self.action_dim, self.n_flow_steps, self.device
            )  # [1, T, A]

        normed_np = normed_chunk[0].cpu().float().numpy()  # [T, A]
        raw_chunk = self.normalizer.denormalize_chunk(normed_np)
        return raw_chunk  # [chunk_size, action_dim]

    @staticmethod
    def _validate_schema(obj: dict, required_keys: list[str]) -> None:
        missing = [k for k in required_keys if k not in obj]
        if missing:
            raise ValueError(f"Schema validation failed. Missing keys: {missing}. Got: {obj}")
