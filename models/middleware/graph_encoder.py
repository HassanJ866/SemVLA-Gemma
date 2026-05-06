"""
Simple bag-of-relations graph encoder.

Encodes a list of [subject, relation, object] triplets as a fixed-size float
vector via one-hot relation counts (L2-normalised). This is the default
implementation as specified; upgrade to a GNN only if parsing F1 transfer
to adapter proves poor.

GRAPH_FEAT_DIM must match the adapter's graph_feat_dim hyperparameter.
"""

import numpy as np
import torch

RELATIONS = [
    "is_left_of",
    "is_right_of",
    "is_above",
    "is_below",
    "is_in_front_of",
    "is_behind",
    "is_on",
    "is_under",
    "is_inside",
    "contains",
]
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
N_RELS = len(RELATIONS)
GRAPH_FEAT_DIM = 64  # zero-padded beyond N_RELS; matches adapter default


def encode_graph(triplets: list[list[str]], dim: int = GRAPH_FEAT_DIM) -> np.ndarray:
    """
    Returns a float32 vector of length `dim`.
    Entries 0..N_RELS-1 are relation counts; rest are zero.
    Vector is L2-normalised.
    """
    vec = np.zeros(dim, dtype=np.float32)
    for triplet in triplets:
        if len(triplet) != 3:
            continue
        rel_id = REL_TO_ID.get(triplet[1], -1)
        if 0 <= rel_id < dim:
            vec[rel_id] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def encode_graph_tensor(triplets: list[list[str]], dim: int = GRAPH_FEAT_DIM,
                         device: str = "cpu") -> torch.Tensor:
    """Returns shape [1, 1, dim] (batch=1, seq_len=1) for direct adapter input."""
    vec = encode_graph(triplets, dim)
    return torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
