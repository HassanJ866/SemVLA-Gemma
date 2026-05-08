"""
SemVLAPolicy — SmolVLA-compatible policy that uses our finetuned Gemma 4 E4B
brain (frozen) as the VLM backbone, with an identical flow-matching action
expert trained on top.

The only structural change vs SmolVLAPolicy:
- SmolVLMWithExpertModel → Gemma4WithExpertModel
- brain_model_path instead of vlm_model_name
- chunk_size default 16 (LIBERO) instead of 50

Everything else — forward(), sample_actions(), denoise_step(), embed_prefix(),
embed_suffix(), loss computation — is identical to SmolVLA.
"""

import math
from collections import deque

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype
from lerobot.utils.import_utils import require_package

from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_semvla import SemVLAConfig
from .gemma4_with_expert import Gemma4WithExpertModel


# ── Utility functions (copied verbatim from modeling_smolvla.py) ──────────────

def create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("time must be shape (batch_size,)")
    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def make_att_2d_masks(pad_masks, att_masks):
    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d & pad_2d


def resize_with_pad(img, width, height, pad_value=-1):
    if img.ndim != 4:
        raise ValueError(f"Expected (b,c,h,w), got {img.shape}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    rh = int(cur_h / ratio)
    rw = int(cur_w / ratio)
    img = F.interpolate(img, size=(rh, rw), mode="bilinear", align_corners=False)
    pad_h = max(0, int(height - rh))
    pad_w = max(0, int(width - rw))
    return F.pad(img, (pad_w, 0, pad_h, 0), value=pad_value)


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    out = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    out[..., :vector.shape[-1]] = vector
    return out


def pad_tensor(tensor, max_len, pad_value=0):
    b, d = tensor.shape[:2]
    out = torch.full((b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device)
    out[:, :d] = tensor
    return out


# ── Main policy ───────────────────────────────────────────────────────────────

class SemVLAPolicy(PreTrainedPolicy):
    """LeRobot policy that wraps SemVLAFlowMatching for training and inference."""

    config_class = SemVLAConfig
    name = "semvla"

    def __init__(self, config: SemVLAConfig, **kwargs):
        require_package("transformers", extra="smolvla")
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = SemVLAFlowMatching(config)
        self.reset()

    def reset(self):
        self._queues = {ACTION: torch.nn.modules.container.ModuleList.__new__(
            type("_Deque", (), {})
        )}
        # Use a plain deque — identical to SmolVLAPolicy
        from collections import deque
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}

    def get_optim_params(self):
        return self.parameters()

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])
        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        self.eval()
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        return self._get_action_chunk(batch, noise)

    def _get_action_chunk(self, batch, noise=None):
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)
        images, img_masks, img_position_ids = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.model.sample_actions(images, img_masks, img_position_ids, lang_tokens, lang_masks, state, noise=noise)
        original_action_dim = self.config.action_feature.shape[0]
        return actions[:, :, :original_action_dim]

    # ── Training ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict[str, Tensor], noise=None, time=None, reduction="mean"):
        images, img_masks, img_position_ids = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        losses = self.model.forward(images, img_masks, img_position_ids, lang_tokens, lang_masks, state, actions, noise, time)
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        loss_dict = {"losses_after_forward": losses.clone().mean().item()}

        if actions_is_pad is not None:
            losses = losses * (~actions_is_pad).unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        losses = losses[:, :, :self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            if actions_is_pad is None:
                per_sample = losses.mean(dim=(1, 2))
            else:
                n_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample = losses.sum(dim=(1, 2)) / n_valid
            loss_dict["loss"] = per_sample.mean().item()
            return per_sample, loss_dict

        if actions_is_pad is None:
            loss = losses.mean()
        else:
            n_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
            loss = losses.sum() / n_valid
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    # ── Preprocessing helpers ─────────────────────────────────────────────────

    def prepare_images(self, batch):
        images, img_masks, img_position_ids = [], [], []
        present = [k for k in self.config.image_features if k in batch]
        missing = [k for k in self.config.image_features if k not in batch]
        if not present:
            raise ValueError(f"No image features found in batch. Expected: {list(self.config.image_features)}")
        for key in present:
            img = batch[key][:, -1] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            img = img * 2.0 - 1.0   # [0,1] → [-1,1] for SigLIP/Gemma4 vision
            bsize = img.shape[0]
            mask = (batch[f"{key}_padding_mask"].bool() if f"{key}_padding_mask" in batch
                    else torch.ones(bsize, dtype=torch.bool, device=img.device))
            
            pos_ids = batch.get(f"{key}_position_ids")
            if pos_ids is None and key == "observation.image":
                pos_ids = batch.get("image_position_ids")
            if pos_ids is not None:
                pos_ids = pos_ids[:, -1] if pos_ids.ndim == 4 else pos_ids
                
            images.append(img)
            img_masks.append(mask)
            img_position_ids.append(pos_ids)
            
        for i in range(min(len(missing), self.config.empty_cameras)):
            images.append(torch.ones_like(images[-1]) * -1)
            img_masks.append(torch.zeros_like(img_masks[-1]))
            img_position_ids.append(None)
        return images, img_masks, img_position_ids

    def prepare_state(self, batch):
        state = batch[OBS_STATE][:, -1] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        return pad_vector(state, self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)


# ── Flow-matching model ───────────────────────────────────────────────────────

class SemVLAFlowMatching(nn.Module):
    """
    Core flow-matching model:
      prefix = [image_embs | lang_embs | state_emb]  → processed by frozen Gemma4
      suffix = [noisy_action_embs | time_emb]         → processed by trainable expert
    """

    def __init__(self, config: SemVLAConfig):
        super().__init__()
        self.config = config

        self.gemma4_with_expert = Gemma4WithExpertModel(
            brain_model_path=config.brain_model_path,
            load_brain_weights=config.load_brain_weights,
            train_expert_only=config.train_expert_only,
            freeze_vision_encoder=config.freeze_vision_encoder,
            attention_mode=config.attention_mode,
            num_expert_layers=config.num_expert_layers,
            self_attn_every_n_layers=config.self_attn_every_n_layers,
            expert_width_multiplier=config.expert_width_multiplier,
        )

        vlm_hidden = self.gemma4_with_expert._text_model.config.hidden_size
        expert_hidden = self.gemma4_with_expert.expert_hidden_size

        self.state_proj = nn.Linear(config.max_state_dim, vlm_hidden)
        self.action_in_proj = nn.Linear(config.max_action_dim, expert_hidden)
        self.action_out_proj = nn.Linear(expert_hidden, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(expert_hidden * 2, expert_hidden)
        self.action_time_mlp_out = nn.Linear(expert_hidden, expert_hidden)

        self.set_requires_grad()
        self.prefix_length = config.prefix_length

    def set_requires_grad(self):
        for p in self.state_proj.parameters():
            p.requires_grad = self.config.train_state_proj

    # ── Noise / time sampling ─────────────────────────────────────────────────

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        beta = torch.distributions.Beta(1.5, 1.0)
        t = beta.sample((bsize,)).to(device=device, dtype=torch.float32)
        return t * 0.999 + 0.001

    # ── Embedding ─────────────────────────────────────────────────────────────

    def embed_prefix(self, images, img_masks, img_position_ids, lang_tokens, lang_masks, state):
        embs, pad_masks, att_masks = [], [], []

        for img, img_mask, img_pos_id in zip(images, img_masks, img_position_ids):
            img_emb = self.gemma4_with_expert.embed_image(img, pixel_position_ids=img_pos_id)
            # Normalize by sqrt(d) — same as SmolVLA
            img_emb = img_emb * (img_emb.shape[-1] ** 0.5)
            bsize, n_patches = img_emb.shape[:2]
            mask = img_mask[:, None].expand(bsize, n_patches)
            embs.append(img_emb)
            pad_masks.append(mask)
            att_masks += [0] * n_patches

        lang_emb = self.gemma4_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * lang_emb.shape[1]

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :]
        bsize, device = state_emb.shape[0], state_emb.device
        embs.append(state_emb)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [1]   # state separates prefix from suffix in attention

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)[None, :].expand(bsize, -1)

        # Pad to fixed prefix length if configured
        if self.prefix_length > 0 and pad_masks.shape[1] < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        action_emb = self.action_in_proj(noisy_actions)
        device, bsize, dtype = action_emb.device, action_emb.shape[0], action_emb.dtype

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.gemma4_with_expert.expert_hidden_size,
            self.config.min_period, self.config.max_period, device=device,
        ).to(dtype)
        time_emb = time_emb[:, None, :].expand_as(action_emb)

        fused = torch.cat([action_emb, time_emb], dim=2)
        fused = self.action_time_mlp_in(fused)
        fused = F.silu(fused)
        fused = self.action_time_mlp_out(fused)

        pad_mask = torch.ones(bsize, fused.shape[1], dtype=torch.bool, device=device)
        att_mask = torch.ones(bsize, self.config.chunk_size, dtype=fused.dtype, device=device)
        return fused, pad_mask, att_mask

    # ── Training forward ──────────────────────────────────────────────────────

    def forward(self, images, img_masks, img_position_ids, lang_tokens, lang_masks, state, actions, noise=None, time=None):
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        t = time[:, None, None]
        x_t = t * noise + (1 - t) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            images, img_masks, img_position_ids, lang_tokens, lang_masks, state
        )
        suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad, suffix_pad], dim=1)
        att_masks = torch.cat([prefix_att, suffix_att], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.gemma4_with_expert.forward(
            attention_mask=att_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size:].to(torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    # ── Inference ─────────────────────────────────────────────────────────────

    def sample_actions(self, images, img_masks, img_position_ids, lang_tokens, lang_masks, state, noise=None):
        bsize, device = state.shape[0], state.device
        if noise is None:
            noise = self.sample_noise((bsize, self.config.chunk_size, self.config.max_action_dim), device)

        prefix_embs, prefix_pad, prefix_att = self.embed_prefix(
            images, img_masks, img_position_ids, lang_tokens, lang_masks, state
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad, prefix_att)
        prefix_pos = torch.cumsum(prefix_pad, dim=1) - 1

        _, past_kv = self.gemma4_with_expert.forward(
            attention_mask=prefix_att_2d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        x_t = noise
        dt = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_t = self.denoise_step(x_t, t_tensor, prefix_pad, past_kv)
            x_t = x_t + dt * v_t

        return x_t

    def denoise_step(self, x_t, timestep, prefix_pad_masks, past_key_values):
        suffix_embs, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
        bsize = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad.shape[1]

        prefix_2d = prefix_pad_masks[:, None, :].expand(bsize, suffix_len, prefix_len)
        suffix_2d = make_att_2d_masks(suffix_pad, suffix_att)
        full_att = torch.cat([prefix_2d, suffix_2d], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1

        outputs, _ = self.gemma4_with_expert.forward(
            attention_mask=full_att,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        out = outputs[1][:, -self.config.chunk_size:].to(torch.float32)
        return self.action_out_proj(out)
