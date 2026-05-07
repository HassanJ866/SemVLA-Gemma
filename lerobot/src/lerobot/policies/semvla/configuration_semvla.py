from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES


@PreTrainedConfig.register_subclass("semvla")
@dataclass
class SemVLAConfig(PreTrainedConfig):
    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 16          # LIBERO default (vs 50 in SmolVLA)
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state/action vectors will be zero-padded to these dims
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing — resize+pad to square before SigLIP / Gemma4 vision
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    empty_cameras: int = 0

    # Tokenizer — use Gemma4 processor tokenizer
    tokenizer_max_length: int = 64   # slightly longer than SmolVLA (48) for scene-graph prompts

    # Flow-matching denoising steps at inference
    num_steps: int = 10

    use_cache: bool = True

    # ── Gemma4 brain (frozen) ────────────────────────────────────────────────
    # Path to the finetuned Gemma4 checkpoint (local path or HF repo id).
    # Set to the final brain checkpoint after Phase 1 training.
    brain_model_path: str = "ckpts/brain_phase1/final"

    # Whether to load brain weights (True) or init from config only (False, for dry-run)
    load_brain_weights: bool = True

    # ── Action expert ────────────────────────────────────────────────────────
    # Expert hidden size as a fraction of Gemma4's text hidden size (2048 for E4B)
    expert_width_multiplier: float = 0.5

    # How many VLM layers the expert mirrors; must divide num_vlm_layers (42)
    num_expert_layers: int = 7

    # Attention mode between VLM prefix and expert suffix
    # "cross_attn": expert Q attends to VLM K/V (recommended — expert stays small)
    # "self_attn":  full joint self-attention (heavier, but simpler)
    attention_mode: str = "cross_attn"

    # Interleave a self-attention layer in the expert every N layers (cross_attn mode only)
    self_attn_every_n_layers: int = 2

    # Sine-cosine timestep encoding sensitivity range
    min_period: float = 4e-3
    max_period: float = 4.0

    # ── Training flags ───────────────────────────────────────────────────────
    freeze_vision_encoder: bool = True   # freeze Gemma4 vision tower
    train_expert_only: bool = True       # freeze entire Gemma4 brain, train expert only
    train_state_proj: bool = True

    # ── Optimizer / scheduler presets ───────────────────────────────────────
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0

    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 2.5e-6

    # Prefix-length padding (keep at -1 = auto)
    prefix_length: int = -1

    pad_language_to: str = "longest"

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})"
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            self.input_features[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
