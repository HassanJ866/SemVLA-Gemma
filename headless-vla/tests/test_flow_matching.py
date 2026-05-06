"""
Tests for the adapter architecture and flow matching loss/inference.
No GPU required (runs on CPU).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.adapter.architecture import (
    SemanticActionAdapter,
    CrossAttnBlock,
    CausalSelfAttnBlock,
    SinusoidalTimestepEmbedding,
)
from models.adapter.flow_matching import (
    flow_matching_loss,
    flow_matching_inference,
    sample_beta_tau,
)


def _make_adapter(action_dim=7, chunk_size=4, hidden=32, n_heads=2,
                  n_blocks=4, graph_feat_dim=16, state_dim=9):
    return SemanticActionAdapter(
        action_dim=action_dim, chunk_size=chunk_size, hidden=hidden,
        n_heads=n_heads, n_blocks=n_blocks, graph_feat_dim=graph_feat_dim,
        state_dim=state_dim,
    )


TASK_EMBED_DIM = 384


def _make_batch(B=2, chunk_size=4, action_dim=7, graph_feat_dim=16, state_dim=9):
    task_emb = torch.randn(B, TASK_EMBED_DIM)
    graph    = torch.randn(B, 1, graph_feat_dim)
    state    = torch.randn(B, state_dim)
    actions  = torch.randn(B, chunk_size, action_dim)
    tau      = torch.rand(B)
    return task_emb, graph, state, actions, tau


def test_adapter_forward_shape():
    adapter = _make_adapter()
    task_emb, graph, state, actions, tau = _make_batch()
    out = adapter(task_emb, graph, state, actions, tau)
    assert out.shape == (2, 4, 7), f"Expected (2,4,7), got {out.shape}"


def test_sinusoidal_embedding():
    emb = SinusoidalTimestepEmbedding(32)
    tau = torch.rand(4)
    out = emb(tau)
    assert out.shape == (4, 32)
    assert not torch.isnan(out).any()


def test_cross_attn_block():
    block = CrossAttnBlock(32, 2)
    x   = torch.randn(2, 4, 32)
    ctx = torch.randn(2, 6, 32)
    out = block(x, ctx)
    assert out.shape == (2, 4, 32)


def test_causal_self_attn_block():
    block = CausalSelfAttnBlock(32, 2)
    x = torch.randn(2, 4, 32)
    out = block(x)
    assert out.shape == (2, 4, 32)


def test_flow_matching_loss_scalar():
    adapter = _make_adapter()
    task_emb, graph, state, actions, _ = _make_batch()
    loss = flow_matching_loss(adapter, task_emb, graph, state, actions)
    assert loss.ndim == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_flow_matching_loss_backward():
    adapter = _make_adapter()
    task_emb, graph, state, actions, _ = _make_batch()
    loss = flow_matching_loss(adapter, task_emb, graph, state, actions)
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in adapter.parameters()
                  if p.grad is not None]
    assert len(grad_norms) > 0
    assert all(g == g for g in grad_norms)  # no NaN grads


def test_flow_matching_inference_shape():
    adapter = _make_adapter()
    task_emb, graph, state, _, _ = _make_batch(B=2)
    with torch.no_grad():
        chunk = flow_matching_inference(adapter, task_emb, graph, state,
                                        chunk_size=4, action_dim=7, n_steps=5)
    assert chunk.shape == (2, 4, 7)
    assert not torch.isnan(chunk).any()


def test_sample_beta_tau():
    tau = sample_beta_tau(1000, alpha=1.5, beta=1.0)
    assert tau.shape == (1000,)
    assert (tau >= 0).all() and (tau <= 1).all()
    # Beta(1.5, 1.0) biases toward 1; mean should be alpha/(alpha+beta) = 0.6
    assert abs(tau.mean().item() - 0.6) < 0.05


def test_adapter_save_load(tmp_path):
    adapter = _make_adapter(action_dim=7, chunk_size=4)
    adapter.save(str(tmp_path))

    import json
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg["action_dim"] == 7
    assert cfg["chunk_size"] == 4

    # reload
    loaded = SemanticActionAdapter(**cfg)
    loaded.load_state_dict(torch.load(tmp_path / "adapter.pt", weights_only=True))
    task_emb, graph, state, actions, tau = _make_batch()
    with torch.no_grad():
        out = loaded(task_emb, graph, state, actions, tau)
    assert out.shape == (2, 4, 7)


if __name__ == "__main__":
    test_sinusoidal_embedding()
    test_cross_attn_block()
    test_causal_self_attn_block()
    test_adapter_forward_shape()
    test_flow_matching_loss_scalar()
    test_flow_matching_loss_backward()
    test_flow_matching_inference_shape()
    test_sample_beta_tau()
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        test_adapter_save_load(Path(d))
    print("All flow matching tests passed.")
