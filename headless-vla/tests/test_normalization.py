"""
Tests for action normalisation / denormalisation stats computation.
No GPU required.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.middleware.normalize import ActionNormalizer, compute_and_save_stats


def _write_dummy_jsonl(path: Path, n_records: int = 50, action_dim: int = 7,
                       chunk_size: int = 16):
    rng = np.random.default_rng(42)
    with open(path, "w") as f:
        for _ in range(n_records):
            chunk = rng.uniform(-0.05, 0.05, (chunk_size, action_dim)).tolist()
            record = {
                "semantic_ids": [0, 3, 5, 8],
                "graph_feats": [0.0] * 64,
                "proprio": [0.0] * 9,
                "action_chunk": chunk,
            }
            f.write(json.dumps(record) + "\n")


def test_compute_stats_creates_file():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        jsonl = d / "adapter.jsonl"
        stats_path = d / "action_stats.json"
        _write_dummy_jsonl(jsonl)
        stats = compute_and_save_stats(str(jsonl), str(stats_path), action_dim=7)
        assert stats_path.exists()
        assert stats["action_dim"] == 7
        assert len(stats["min"]) == 7
        assert len(stats["max"]) == 7
        assert stats["n_steps"] == 50 * 16


def test_normalize_denormalize_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        jsonl = d / "adapter.jsonl"
        stats_path = d / "action_stats.json"
        _write_dummy_jsonl(jsonl)
        compute_and_save_stats(str(jsonl), str(stats_path), action_dim=7)

        norm = ActionNormalizer(str(stats_path))
        rng = np.random.default_rng(0)
        raw = rng.uniform(-0.05, 0.05, (16, 7)).astype(np.float32)

        normed = norm.normalize_chunk(raw)
        recovered = norm.denormalize_chunk(normed)
        np.testing.assert_allclose(recovered, raw, atol=1e-5,
                                   err_msg="Normalize/denormalize roundtrip failed")


def test_normalized_range():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        jsonl = d / "adapter.jsonl"
        stats_path = d / "action_stats.json"
        _write_dummy_jsonl(jsonl)
        compute_and_save_stats(str(jsonl), str(stats_path), action_dim=7)

        norm = ActionNormalizer(str(stats_path))
        # Training data values should normalise close to [-1, 1]
        raw_min = np.array(json.loads(stats_path.read_text())["min"], dtype=np.float32)
        raw_max = np.array(json.loads(stats_path.read_text())["max"], dtype=np.float32)
        normed_min = norm.normalize(raw_min)
        normed_max = norm.normalize(raw_max)
        np.testing.assert_allclose(normed_min, np.full(7, -1.0), atol=1e-5)
        np.testing.assert_allclose(normed_max, np.full(7,  1.0), atol=1e-5)


def test_missing_stats_raises():
    try:
        ActionNormalizer("/nonexistent/path/stats.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    test_compute_stats_creates_file()
    test_normalize_denormalize_roundtrip()
    test_normalized_range()
    test_missing_stats_raises()
    print("All normalization tests passed.")
