"""
Gemma4WithExpertModel — mirrors SmolVLMWithExpertModel but loads our finetuned
Gemma 4 E4B brain (frozen) as the VLM backbone, keeping the action-expert
cross-attention architecture identical.

Key differences vs SmolVLMWithExpertModel:
- Loads Gemma4ForConditionalGeneration (Unsloth/transformers) instead of SmolVLM
- Gemma4 hidden size: 2048 (E4B), vs SmolVLM 576
- Vision connector: Gemma4 uses its own vision projector (not SmolVLM's connector)
- embed_image() wraps the Gemma4 vision tower + multi-modal projector
- embed_language_tokens() uses Gemma4's text embedding table
- The rest of the forward pass (cross-attn expert loop) is identical
"""

import copy

import torch
from torch import nn

from lerobot.utils.import_utils import _transformers_available, require_package

if _transformers_available:
    from transformers import AutoConfig, AutoModel, AutoProcessor
else:
    AutoConfig = None
    AutoModel = None
    AutoProcessor = None


def apply_rope(x, positions, max_wavelength=10_000):
    """RoPE: positions [B, L], x [B, L, H, D]."""
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)
    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength ** freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :]
    radians = radians[..., None, :]
    sin = torch.sin(radians)
    cos = torch.cos(radians)
    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin
    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class Gemma4WithExpertModel(nn.Module):
    """
    Frozen Gemma4 brain + trainable action expert.

    The VLM (Gemma4) is always frozen when train_expert_only=True.
    The action expert is a smaller Llama-style transformer that attends
    cross-attention onto the VLM's key/value cache.
    """

    def __init__(
        self,
        brain_model_path: str,
        load_brain_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = True,
        attention_mode: str = "cross_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = 2,
        expert_width_multiplier: float = 0.5,
    ):
        super().__init__()
        require_package("transformers", extra="smolvla")

        # ── Load Gemma4 brain ────────────────────────────────────────────────
        if load_brain_weights:
            print(f"Loading Gemma4 brain from {brain_model_path} …")
            from transformers import AutoModelForImageTextToText
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                brain_model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            from transformers import AutoModelForImageTextToText
            # If this is a LoRA adapter dir, get base model path from adapter_config.json
            import os, json
            adapter_cfg_path = os.path.join(brain_model_path, "adapter_config.json")
            if os.path.exists(adapter_cfg_path):
                with open(adapter_cfg_path) as f:
                    adapter_cfg = json.load(f)
                base_path = adapter_cfg["base_model_name_or_path"]
            else:
                base_path = brain_model_path
            cfg = AutoConfig.from_pretrained(base_path, trust_remote_code=True)
            self.vlm = AutoModelForImageTextToText.from_config(cfg)

        self.processor = AutoProcessor.from_pretrained(brain_model_path, trust_remote_code=True)

        # Resolve the text model (Gemma4 nests it differently from SmolVLM)
        # Gemma4ForConditionalGeneration → .model (Gemma4Model) → .language_model (Gemma2Model)
        # or directly .model.language_model depending on transformers version.
        # We discover the right attribute at runtime.
        self._text_model = self._resolve_text_model()
        self._vision_model = self._resolve_vision_model()

        # Trim VLM layers if requested
        text_layers = self._text_model.layers
        if num_vlm_layers > 0:
            print(f"Trimming VLM to {num_vlm_layers} layers …")
            self._text_model.layers = text_layers[:num_vlm_layers]
        self.num_vlm_layers = len(self._text_model.layers)

        vlm_hidden_size = self._text_model.config.hidden_size

        # ── Build action expert (smaller Llama-style model) ──────────────────
        # We borrow SmolVLM2-500M's text config as the expert backbone config
        # because it's Llama-compatible and small. Alternatively, we could use
        # Gemma2's text config scaled down — here we use Gemma4's own text
        # config with reduced hidden size for compatibility.
        expert_config = copy.deepcopy(self._text_model.config)
        expert_config.hidden_size = int(vlm_hidden_size * expert_width_multiplier)
        expert_config.intermediate_size = get_intermediate_size(expert_config.hidden_size)
        expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert self.num_vlm_layers % num_expert_layers == 0, (
                f"num_vlm_layers ({self.num_vlm_layers}) must be divisible by "
                f"num_expert_layers ({num_expert_layers})"
            )
            expert_config.num_hidden_layers = num_expert_layers

        self.lm_expert = AutoModel.from_config(expert_config)
        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers

        # Reshape k/v projections for cross-attention (VLM dim → expert dim)
        if "cross" in attention_mode:
            for layer_idx in range(len(self.lm_expert.layers)):
                if self_attn_every_n_layers > 0 and layer_idx % self_attn_every_n_layers == 0:
                    continue
                attn = self.lm_expert.layers[layer_idx].self_attn
                kv_in = (
                    self._text_model.config.num_key_value_heads
                    * self._text_model.config.head_dim
                )
                kv_out = (
                    expert_config.num_key_value_heads
                    * expert_config.head_dim
                )
                bias = getattr(expert_config, "attention_bias", False)
                attn.k_proj = nn.Linear(kv_in, kv_out, bias=bias)
                attn.v_proj = nn.Linear(kv_in, kv_out, bias=bias)

        # Remove unused embedding table from expert
        self.lm_expert.embed_tokens = None

        # VLM head config
        self.vlm_num_attention_heads = self._text_model.config.num_attention_heads
        self.vlm_num_key_value_heads = self._text_model.config.num_key_value_heads
        self.vlm_head_dim = self._text_model.config.head_dim

        # Expert head config (scaled-down hidden, same num_heads → smaller head_dim)
        self.expert_num_attention_heads = expert_config.num_attention_heads
        self.expert_num_key_value_heads = expert_config.num_key_value_heads
        self.expert_head_dim = expert_config.hidden_size // expert_config.num_attention_heads

        self.expert_hidden_size = expert_config.hidden_size
        self.attention_mode = attention_mode
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only

        # Expose config for downstream use (mirrors SmolVLMWithExpertModel.config)
        self.config = self.vlm.config

        self.set_requires_grad()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _resolve_text_model(self):
        """Return the bare Gemma4 text transformer (the .layers attribute must exist)."""
        vlm_model = self.vlm
        # Try common nesting patterns
        for attr_path in [
            "model.language_model",   # Gemma4ForConditionalGeneration in some versions
            "model.text_model",       # SmolVLM-style
            "language_model",         # top-level
            "model",                  # fallback
        ]:
            obj = vlm_model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if hasattr(obj, "layers"):
                    return obj
            except AttributeError:
                continue
        raise RuntimeError(
            "Cannot locate the text transformer inside the Gemma4 model. "
            "Attributes tried: model.language_model, model.text_model, language_model, model"
        )

    def _resolve_vision_model(self):
        """Return the vision encoder module."""
        for attr_path in [
            "model.vision_tower",
            "vision_model",
            "model.vision_model",
        ]:
            obj = self.vlm
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Cannot locate the vision encoder inside the Gemma4 model.")

    def _get_multimodal_projector(self):
        """Return the vision→text projection module."""
        for attr_path in [
            "model.multi_modal_projector",
            "model.connector",
            "multi_modal_projector",
        ]:
            obj = self.vlm
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        return None

    # ── Gradient / training mode ─────────────────────────────────────────────

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self._vision_model.eval()
            for p in self._vision_model.parameters():
                p.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for p in self.vlm.parameters():
                p.requires_grad = False
        for name, p in self.lm_expert.named_parameters():
            if "lm_head" in name:
                p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self._vision_model.eval()
        if self.train_expert_only:
            self.vlm.eval()

    # ── Embedding helpers ────────────────────────────────────────────────────

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Run image through Gemma4 vision tower + multi-modal projector.
        Returns (B, num_patches, vlm_hidden_size).
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor of shape (B, C, H, W), got {tuple(image.shape)}")

        # Gemma4 vision tower expects patchified pixel_values + pixel_position_ids.
        # Our policy path prepares images in [-1, 1], so convert back to [0, 1]
        # before handing off to Gemma4's image processor for patchification.
        image_for_processor = image
        if image_for_processor.dtype != torch.float32:
            image_for_processor = image_for_processor.float()
        if torch.min(image_for_processor).item() < 0:
            image_for_processor = (image_for_processor + 1.0) / 2.0
        image_for_processor = image_for_processor.clamp(0.0, 1.0)

        processor_inputs = self.processor.image_processor(
            images=[img for img in image_for_processor],
            return_tensors="pt",
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
        )
        pixel_values = processor_inputs["pixel_values"]
        pixel_position_ids = processor_inputs["image_position_ids"]

        if pixel_values.ndim != 3:
            raise ValueError(
                f"Expected patchified pixel_values with shape (B, N, patch_dim), got {tuple(pixel_values.shape)}"
            )
        if pixel_position_ids.ndim != 3 or pixel_position_ids.shape[-1] != 2:
            raise ValueError(
                "Expected image_position_ids with shape (B, N, 2), "
                f"got {tuple(pixel_position_ids.shape)}"
            )
        if pixel_values.shape[:2] != pixel_position_ids.shape[:2]:
            raise ValueError(
                "Mismatch between pixel_values and image_position_ids token dims: "
                f"{tuple(pixel_values.shape[:2])} vs {tuple(pixel_position_ids.shape[:2])}"
            )

        expected_patch_dim = 3 * (self._vision_model.config.patch_size ** 2)
        if pixel_values.shape[-1] != expected_patch_dim:
            raise ValueError(
                f"Unexpected patch dim {pixel_values.shape[-1]} (expected {expected_patch_dim}). "
                "Check image preprocessing/patchification."
            )

        vision_dtype = self._vision_model.dtype if hasattr(self._vision_model, "dtype") else pixel_values.dtype
        vision_out = self._vision_model(
            pixel_values=pixel_values.to(device=image.device, dtype=vision_dtype),
            pixel_position_ids=pixel_position_ids.to(device=image.device, dtype=torch.long),
        )
        # last_hidden_state shape: (B, num_patches, vision_hidden)
        image_features = vision_out.last_hidden_state

        projector = self._get_multimodal_projector()
        if projector is not None:
            image_features = projector(image_features)

        return image_features

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed token IDs using Gemma4's text embedding table."""
        return self._text_model.get_input_embeddings()(tokens)

    # ── Attention layers (identical logic to SmolVLMWithExpertModel) ─────────

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        use_cache=True,
        fill_kv_cache=True,
        past_key_values=None,
    ):
        # Self-attention: VLM and expert attend jointly to their concatenated tokens.
        # Each model segment uses its own head_dim from the layer config.
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape))
            key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape))
            value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape))

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)

        seq_len = query_states.shape[1]
        _pos = position_ids[:, :seq_len] if seq_len < position_ids.shape[1] else position_ids
        _mask = attention_mask[:, :seq_len, :seq_len] if seq_len < attention_mask.shape[1] else attention_mask

        query_states = apply_rope(query_states, _pos)
        key_states = apply_rope(key_states, _pos)

        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        # Use VLM head config for the joint self-attention pass
        att_output = self._eager_attention(
            _mask, batch_size, self.vlm_head_dim,
            self.vlm_num_attention_heads, self.vlm_num_key_value_heads,
            query_states, key_states, value_states,
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        use_cache=True,
        fill_kv_cache=True,
        past_key_values=None,
    ):
        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache)

        if len(inputs_embeds) == 2 and not past_key_values:
            seq_len = inputs_embeds[0].shape[1]
            pos_id = position_ids[:, :seq_len]
            expert_position_id = position_ids[:, seq_len:]
            prefix_mask = attention_mask[:, :seq_len, :seq_len]
            layer = model_layers[0][layer_idx]
            hs = layer.input_layernorm(inputs_embeds[0])
            shape = (*hs.shape[:-1], -1, layer.self_attn.head_dim)
            hs = hs.to(dtype=layer.self_attn.q_proj.weight.dtype)
            q = layer.self_attn.q_proj(hs).view(shape)
            k = layer.self_attn.k_proj(hs).view(shape)
            v = layer.self_attn.v_proj(hs).view(shape)
            q = apply_rope(q, pos_id)
            key_states = apply_rope(k, pos_id)
            value_states = v
            att_outputs.append(self._eager_attention(
                prefix_mask, batch_size, self.vlm_head_dim,
                self.vlm_num_attention_heads, self.vlm_num_key_value_heads,
                q, key_states, value_states,
            ))
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}
        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {"key_states": key_states, "value_states": value_states}
            else:
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            ehs = expert_layer.input_layernorm(inputs_embeds[1])
            eshape = (*ehs.shape[:-1], -1, expert_layer.self_attn.head_dim)
            ehs = ehs.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            eq = expert_layer.self_attn.q_proj(ehs).view(eshape)
            # k/v_proj on expert cross-attn layers project FROM VLM KV dims TO expert KV dims
            _k = key_states.to(expert_layer.self_attn.k_proj.weight.dtype).view(*key_states.shape[:2], -1)
            ek = expert_layer.self_attn.k_proj(_k).view(*_k.shape[:-1], -1, expert_layer.self_attn.head_dim)
            _v = value_states.to(expert_layer.self_attn.v_proj.weight.dtype).view(*value_states.shape[:2], -1)
            ev = expert_layer.self_attn.v_proj(_v).view(*_v.shape[:-1], -1, expert_layer.self_attn.head_dim)
            expert_position_id = expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            emask = attention_mask[:, -inputs_embeds[1].shape[1]:, :ek.shape[1]]
            eq = apply_rope(eq, expert_position_id)
            att_outputs.append(self._eager_attention(
                emask, batch_size, self.expert_head_dim,
                self.expert_num_attention_heads, self.expert_num_key_value_heads,
                eq, ek, ev,
            ))
        else:
            att_outputs.append(None)

        return att_outputs, past_key_values

    def get_model_layers(self, models):
        vlm_layers, expert_layers = [], []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layers.append(None)
            else:
                idx = i // multiple_of if multiple_of > 0 else i
                expert_layers.append(models[1].layers[idx])
            vlm_layers.append(models[0].layers[i])
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        fill_kv_cache=None,
    ):
        models = [self._text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)

        batch_size = None
        for hs in inputs_embeds:
            if hs is not None:
                batch_size = hs.shape[0]
                break

        for layer_idx in range(self.num_vlm_layers):
            use_self_attn = (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            )
            if use_self_attn:
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers, inputs_embeds, layer_idx,
                    position_ids, attention_mask, batch_size,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers, inputs_embeds, layer_idx,
                    position_ids, attention_mask, batch_size,
                    use_cache=use_cache, fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )

            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_out = att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]
                    att_out = att_out.to(layer.self_attn.o_proj.weight.dtype)
                    out = layer.self_attn.o_proj(att_out[:, start:end])
                    out = out + hidden_states
                    after_residual = out.clone()
                    out = layer.post_attention_layernorm(out)
                    out = layer.mlp(out)
                    out = out + after_residual
                    outputs_embeds.append(out)
                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)
            inputs_embeds = outputs_embeds

        # Final norm
        final_embeds = []
        for i, hs in enumerate(inputs_embeds):
            if hs is not None:
                final_embeds.append(models[i].norm(hs))
            else:
                final_embeds.append(None)
        return final_embeds, past_key_values

    def _eager_attention(self, attention_mask, batch_size, head_dim, num_heads, num_kv_heads, query_states, key_states, value_states):
        num_groups = num_heads // num_kv_heads
        seq_len = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_groups, head_dim)
        key_states = key_states.reshape(batch_size, seq_len, num_kv_heads * num_groups, head_dim)
        value_states = value_states[:, :, :, None, :].expand(batch_size, seq_len, num_kv_heads, num_groups, head_dim)
        value_states = value_states.reshape(batch_size, seq_len, num_kv_heads * num_groups, head_dim)

        query_states = query_states.to(torch.float32).transpose(1, 2)
        key_states = key_states.to(torch.float32).transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        weights = torch.matmul(query_states, key_states.transpose(2, 3)) * (head_dim ** -0.5)
        big_neg = torch.finfo(weights.dtype).min
        weights = torch.where(attention_mask[:, None, :, :], weights, big_neg)
        probs = torch.nn.functional.softmax(weights, dim=-1).to(value_states.dtype)
        out = torch.matmul(probs, value_states).permute(0, 2, 1, 3)
        out = out.reshape(batch_size, -1, num_kv_heads * num_groups * head_dim)
        return out
