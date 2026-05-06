"""
Tests for middleware: schema validation, graph encoding, and safe-stop fallback.
No GPU required.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.middleware.graph_encoder import encode_graph, encode_graph_tensor, GRAPH_FEAT_DIM


# ── graph encoder tests ────────────────────────────────────────────────────────

def test_graph_encoder_empty():
    feat = encode_graph([], dim=GRAPH_FEAT_DIM)
    assert feat.shape == (GRAPH_FEAT_DIM,)
    assert np.allclose(feat, 0.0)


def test_graph_encoder_known_relation():
    triplets = [["A", "is_left_of", "B"]]
    feat = encode_graph(triplets, dim=GRAPH_FEAT_DIM)
    # is_left_of is index 0 → feat[0] should be 1.0 after L2 normalisation (single entry)
    assert feat[0] == 1.0
    assert np.allclose(np.linalg.norm(feat), 1.0, atol=1e-5)


def test_graph_encoder_multiple_relations():
    triplets = [
        ["A", "is_left_of", "B"],
        ["B", "is_right_of", "A"],
        ["A", "is_on", "C"],
    ]
    feat = encode_graph(triplets, dim=GRAPH_FEAT_DIM)
    assert np.linalg.norm(feat) > 0
    assert np.allclose(np.linalg.norm(feat), 1.0, atol=1e-5)


def test_graph_encoder_unknown_relation():
    triplets = [["A", "UNKNOWN_RELATION", "B"]]
    feat = encode_graph(triplets, dim=GRAPH_FEAT_DIM)
    assert np.allclose(feat, 0.0)


def test_graph_encoder_tensor_shape():
    triplets = [["A", "is_above", "B"]]
    tensor = encode_graph_tensor(triplets, dim=GRAPH_FEAT_DIM)
    assert tensor.shape == (1, 1, GRAPH_FEAT_DIM)
    assert tensor.dtype == torch.float32


# ── chain schema validation tests ─────────────────────────────────────────────

def test_chain_validate_schema_passes():
    from models.middleware.chain import InferenceChain
    InferenceChain._validate_schema({"task": "pick up the bowl"}, ["task"])


def test_chain_validate_schema_fails_missing_key():
    from models.middleware.chain import InferenceChain
    try:
        InferenceChain._validate_schema({}, ["task"])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── safe-stop action shape test ────────────────────────────────────────────────

def test_safe_stop_action_shape():
    from models.middleware.chain import SAFE_STOP_ACTION
    assert SAFE_STOP_ACTION.shape == (7,)
    assert np.allclose(SAFE_STOP_ACTION, 0.0)


if __name__ == "__main__":
    test_graph_encoder_empty()
    test_graph_encoder_known_relation()
    test_graph_encoder_multiple_relations()
    test_graph_encoder_unknown_relation()
    test_graph_encoder_tensor_shape()
    test_chain_validate_schema_passes()
    test_chain_validate_schema_fails_missing_key()
    test_safe_stop_action_shape()
    print("All middleware / safe-stop tests passed.")
