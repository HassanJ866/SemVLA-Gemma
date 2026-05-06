"""
Tests for middleware: schema validation, enum encoding, graph encoding,
and safe-stop fallback behaviour.
No GPU required.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.middleware.enums import (
    semantic_action_to_ids,
    ids_to_semantic_action,
    SAFE_STOP_IDS,
    VOCAB_SIZE,
    AXIS_OFFSET, DIR_OFFSET, MAG_OFFSET, GRIP_OFFSET,
)
from models.middleware.graph_encoder import encode_graph, encode_graph_tensor, GRAPH_FEAT_DIM


# ── enum tests ─────────────────────────────────────────────────────────────────

def test_safe_stop_ids_valid():
    assert len(SAFE_STOP_IDS) == 4
    for i in SAFE_STOP_IDS:
        assert 0 <= i < VOCAB_SIZE, f"ID {i} out of vocab range [0, {VOCAB_SIZE})"


def test_all_enum_ids_in_vocab():
    from models.middleware.enums import AXIS_TO_ID, DIR_TO_ID, MAG_TO_ID, GRIP_TO_ID
    for v in AXIS_TO_ID.values():
        assert 0 <= v + AXIS_OFFSET < VOCAB_SIZE
    for v in DIR_TO_ID.values():
        assert 0 <= v + DIR_OFFSET < VOCAB_SIZE
    for v in MAG_TO_ID.values():
        assert 0 <= v + MAG_OFFSET < VOCAB_SIZE
    for v in GRIP_TO_ID.values():
        assert 0 <= v + GRIP_OFFSET < VOCAB_SIZE


def test_id_offset_no_overlap():
    from models.middleware.enums import AXIS_TO_ID, DIR_TO_ID, MAG_TO_ID, GRIP_TO_ID
    axis_ids = {v + AXIS_OFFSET for v in AXIS_TO_ID.values()}
    dir_ids  = {v + DIR_OFFSET  for v in DIR_TO_ID.values()}
    mag_ids  = {v + MAG_OFFSET  for v in MAG_TO_ID.values()}
    grip_ids = {v + GRIP_OFFSET for v in GRIP_TO_ID.values()}
    # No overlap between positions
    assert not (axis_ids & dir_ids), "Axis and direction IDs overlap"
    assert not (axis_ids & mag_ids), "Axis and magnitude IDs overlap"
    assert not (dir_ids  & mag_ids), "Direction and magnitude IDs overlap"
    assert not (mag_ids  & grip_ids), "Magnitude and gripper IDs overlap"


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
    InferenceChain._validate_schema(
        {"axis": "Z", "direction": "positive", "magnitude": "small", "gripper": "keep"},
        ["axis", "direction", "magnitude", "gripper"]
    )


def test_chain_validate_schema_fails_missing_key():
    from models.middleware.chain import InferenceChain
    try:
        InferenceChain._validate_schema(
            {"axis": "Z", "direction": "positive"},
            ["axis", "direction", "magnitude", "gripper"]
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


# ── safe-stop action shape test ────────────────────────────────────────────────

def test_safe_stop_action_shape():
    from models.middleware.chain import SAFE_STOP_ACTION
    assert SAFE_STOP_ACTION.shape == (7,)
    assert np.allclose(SAFE_STOP_ACTION, 0.0)


if __name__ == "__main__":
    test_safe_stop_ids_valid()
    test_all_enum_ids_in_vocab()
    test_id_offset_no_overlap()
    test_graph_encoder_empty()
    test_graph_encoder_known_relation()
    test_graph_encoder_multiple_relations()
    test_graph_encoder_unknown_relation()
    test_graph_encoder_tensor_shape()
    test_chain_validate_schema_passes()
    test_chain_validate_schema_fails_missing_key()
    test_safe_stop_action_shape()
    print("All middleware / safe-stop tests passed.")
