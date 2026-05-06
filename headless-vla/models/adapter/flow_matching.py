"""
Conditional flow matching loss and Euler denoiser for the action adapter.

Convention (matching SmolVLA):
  tau = 1.0 → clean data (A_clean)
  tau = 0.0 → pure noise
  linear interpolant: A_noisy = tau * A_clean + (1 - tau) * eps
  target velocity:    v*       = eps - A_clean   (flows from clean toward noise)
  Euler update:       A ← A + v / n_steps         (integrates from tau=0 to tau=1)
"""

import torch
import torch.nn.functional as F
from torch import Tensor


# ── tau sampling ───────────────────────────────────────────────────────────────

def sample_beta_tau(batch_size: int, alpha: float = 1.5, beta: float = 1.0,
                    device: str = "cpu") -> Tensor:
    """
    Sample tau ~ Beta(alpha, beta) re-scaled to [0, 1].
    Beta(1.5, 1.0) biases toward tau=1 (near-clean) as recommended.
    Uses PyTorch's Beta distribution.
    """
    dist = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32, device=device),
        torch.tensor(beta,  dtype=torch.float32, device=device),
    )
    return dist.sample((batch_size,))


# ── training loss ──────────────────────────────────────────────────────────────

def flow_matching_loss(
    adapter,
    semantic_ids: Tensor,   # [B, 4]
    graph: Tensor,           # [B, K, G]
    state: Tensor,           # [B, state_dim]
    action_chunk: Tensor,    # [B, T, A]  normalised clean actions
) -> Tensor:
    """
    One training step of conditional flow matching (MSE on velocity field).
    Returns scalar loss.
    """
    B, T, A = action_chunk.shape
    device = action_chunk.device

    eps     = torch.randn_like(action_chunk)                          # [B, T, A]
    tau     = sample_beta_tau(B, device=device)                       # [B]
    tau_bcast = tau[:, None, None]                                    # [B, 1, 1]

    A_noisy = tau_bcast * action_chunk + (1.0 - tau_bcast) * eps      # [B, T, A]
    target_velocity = eps - action_chunk                               # [B, T, A]

    v_pred = adapter(
        semantic_ids=semantic_ids,
        graph=graph,
        state=state,
        noisy_actions=A_noisy,
        tau=tau,
    )
    return F.mse_loss(v_pred, target_velocity)


# ── inference denoiser ─────────────────────────────────────────────────────────

@torch.no_grad()
def flow_matching_inference(
    adapter,
    semantic_ids: Tensor,   # [B, 4]
    graph: Tensor,           # [B, K, G]
    state: Tensor,           # [B, state_dim]
    chunk_size: int,
    action_dim: int,
    n_steps: int = 10,
    device: str = "cpu",
) -> Tensor:
    """
    Euler integration from tau=0 (noise) to tau=1 (clean).
    Returns predicted action chunk [B, chunk_size, action_dim].
    """
    B = semantic_ids.shape[0]
    A = torch.randn(B, chunk_size, action_dim, device=device)

    for k in range(n_steps):
        tau = torch.full((B,), k / n_steps, dtype=torch.float32, device=device)
        v = adapter(
            semantic_ids=semantic_ids,
            graph=graph,
            state=state,
            noisy_actions=A,
            tau=tau,
        )
        A = A + v / n_steps  # Euler step

    return A  # [B, chunk_size, action_dim]
