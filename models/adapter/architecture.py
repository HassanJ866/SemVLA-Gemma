"""
SemanticActionAdapter: flow-matching transformer for continuous action prediction.

Architecture:
  - 4 interleaved blocks: [CrossAttn, CausalSelfAttn, CrossAttn, CausalSelfAttn]
  - Hidden size: 256, heads: 4, FFN: 1024
  - Input: task_emb (1 token, 384-dim mean-pooled Gemma token embedding)
           + graph_feats (1 token) + state (1 token)
           + noisy_action tokens (chunk_size)
  - Output: velocity field [B, chunk_size, action_dim]

~3–8M parameters depending on action_dim and chunk_size.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── sinusoidal timestep embedding ─────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """tau: [B] floats in [0, 1]. Returns [B, dim]."""
        device = tau.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = tau[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


# ── transformer blocks ─────────────────────────────────────────────────────────

class CrossAttnBlock(nn.Module):
    """
    Cross-attention block.
    Q from query_seq; K, V from context_seq.
    Followed by a feed-forward network.
    """

    def __init__(self, hidden: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.norm_q   = nn.LayerNorm(hidden)
        self.norm_ctx = nn.LayerNorm(hidden)
        self.attn     = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm_ff  = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        k = self.norm_ctx(context)
        attn_out, _ = self.attn(q, k, k)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class CausalSelfAttnBlock(nn.Module):
    """
    Causal self-attention block (action tokens attend to earlier action tokens only).
    """

    def __init__(self, hidden: int, n_heads: int, ffn_mult: int = 4):
        super().__init__()
        self.norm_x  = nn.LayerNorm(hidden)
        self.attn    = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden * ffn_mult, hidden),
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # True = masked (forbidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = self._causal_mask(T, x.device)
        normed = self.norm_x(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


# ── adapter ────────────────────────────────────────────────────────────────────

class SemanticActionAdapter(nn.Module):
    """
    Flow-matching adapter that maps:
      (semantic_ids, graph_feats, proprio, noisy_action_chunk, tau)
      → predicted velocity field [B, chunk_size, action_dim]

    Parameters
    ----------
    action_dim      Continuous action dimension (e.g. 7 for Franka).
    chunk_size      Number of timesteps per chunk (default: 16).
    hidden          Transformer hidden size (default: 256).
    n_heads         Attention heads (default: 4).
    n_blocks        Total transformer blocks; alternates CA/SA (default: 4).
    graph_feat_dim  Input dimension of graph feature vector (default: 64).
    state_dim       Proprioceptive state dimension (default: 9).
    ffn_mult        FFN hidden multiplier (default: 4).
    """

    def __init__(
        self,
        action_dim: int,
        chunk_size: int = 16,
        hidden: int = 256,
        n_heads: int = 4,
        n_blocks: int = 4,
        graph_feat_dim: int = 64,
        state_dim: int = 9,
        ffn_mult: int = 4,
        task_embed_dim: int = 384,
    ):
        super().__init__()
        self.action_dim     = action_dim
        self.chunk_size     = chunk_size
        self.hidden         = hidden
        self.task_embed_dim = task_embed_dim

        # input projections
        self.task_proj  = nn.Linear(task_embed_dim, hidden)
        self.graph_proj = nn.Linear(graph_feat_dim, hidden)
        self.state_proj = nn.Linear(state_dim, hidden)
        self.action_in  = nn.Linear(action_dim, hidden)
        self.tau_embed  = SinusoidalTimestepEmbedding(hidden)

        # interleaved CA / SA blocks
        self.blocks = nn.ModuleList([
            CrossAttnBlock(hidden, n_heads, ffn_mult) if i % 2 == 0
            else CausalSelfAttnBlock(hidden, n_heads, ffn_mult)
            for i in range(n_blocks)
        ])

        self.action_out = nn.Linear(hidden, action_dim)

    def forward(
        self,
        task_emb: torch.Tensor,        # [B, task_embed_dim]
        graph: torch.Tensor,           # [B, K, graph_feat_dim]  (K=1 for simple encoder)
        state: torch.Tensor,           # [B, state_dim]
        noisy_actions: torch.Tensor,   # [B, chunk_size, action_dim]
        tau: torch.Tensor,             # [B]
    ) -> torch.Tensor:                 # [B, chunk_size, action_dim]

        # build conditioning context tokens
        task_token   = self.task_proj(task_emb).unsqueeze(1)      # [B, 1, H]
        graph_tokens = self.graph_proj(graph)                     # [B, K, H]
        state_token  = self.state_proj(state).unsqueeze(1)        # [B, 1, H]
        context = torch.cat([task_token, graph_tokens, state_token], dim=1)  # [B, 1+K+1, H]

        # build action query tokens + tau conditioning
        tau_emb = self.tau_embed(tau).unsqueeze(1)                # [B, 1, H]
        x = self.action_in(noisy_actions) + tau_emb               # [B, T, H]  (broadcast)

        # transformer blocks
        for block in self.blocks:
            if isinstance(block, CrossAttnBlock):
                x = block(x, context)
            else:
                x = block(x)

        return self.action_out(x)  # [B, chunk_size, action_dim]

    def save(self, out_dir: str) -> None:
        import json
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), out / "adapter.pt")
        cfg = {
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "hidden": self.hidden,
            "n_heads": self.blocks[0].attn.num_heads
                       if hasattr(self.blocks[0], "attn") else 4,
            "n_blocks": len(self.blocks),
            "graph_feat_dim": self.graph_proj.in_features,
            "state_dim": self.state_proj.in_features,
            "ffn_mult": 4,
            "task_embed_dim": self.task_embed_dim,
        }
        with open(out / "adapter_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
