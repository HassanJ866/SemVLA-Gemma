"""
Action normalisation / denormalisation.

Stats (per-dim min and max) are computed from the full training set BEFORE
adapter training and saved to a JSON file. The adapter is trained on
normalised actions in [-1, 1] space. At inference time the middleware
applies the inverse transform before sending actions to the environment.

Usage — compute stats:
    from models.middleware.normalize import compute_and_save_stats
    compute_and_save_stats("data/splits/adapter_franka.jsonl",
                           "ckpts/franka_7dof/action_stats.json",
                           action_dim=7)

Usage — runtime:
    from models.middleware.normalize import ActionNormalizer
    norm = ActionNormalizer("ckpts/franka_7dof/action_stats.json")
    normed   = norm.normalize(raw_action)
    raw      = norm.denormalize(normed_action)
"""

import json
from pathlib import Path

import numpy as np


class ActionNormalizer:
    def __init__(self, stats_path: str):
        stats_path = Path(stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Action stats not found: {stats_path}. "
                "Run compute_and_save_stats() before adapter training."
            )
        with open(stats_path) as f:
            stats = json.load(f)
        self.min = np.array(stats["min"], dtype=np.float32)
        self.max = np.array(stats["max"], dtype=np.float32)
        self.range = self.max - self.min
        self.range[self.range < 1e-8] = 1.0  # avoid div-by-zero for constant dims

    def normalize(self, action: np.ndarray) -> np.ndarray:
        """Map action from [min, max] to [-1, 1]."""
        return 2.0 * (action - self.min) / self.range - 1.0

    def denormalize(self, normed: np.ndarray) -> np.ndarray:
        """Map action from [-1, 1] back to [min, max]."""
        return (normed + 1.0) / 2.0 * self.range + self.min

    def normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """chunk shape: [T, action_dim]"""
        return self.normalize(chunk)

    def denormalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """chunk shape: [T, action_dim]"""
        return self.denormalize(chunk)


def compute_and_save_stats(adapter_jsonl: str, out_path: str,
                           action_dim: int = 7) -> dict:
    """
    Scan the adapter training JSONL, collect all action chunks, compute
    per-dimension min/max, and write to out_path as JSON.
    This must be called before any adapter training run.
    """
    all_actions = []
    with open(adapter_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunk = rec.get("action_chunk", [])
            if chunk:
                arr = np.array(chunk, dtype=np.float32)  # [T, A]
                all_actions.append(arr)

    if not all_actions:
        raise ValueError(f"No action_chunk data found in {adapter_jsonl}")

    all_actions_np = np.concatenate(all_actions, axis=0)  # [N*T, A]
    # Trim to action_dim in case of extra columns
    all_actions_np = all_actions_np[:, :action_dim]

    stats = {
        "min": all_actions_np.min(axis=0).tolist(),
        "max": all_actions_np.max(axis=0).tolist(),
        "mean": all_actions_np.mean(axis=0).tolist(),
        "std": all_actions_np.std(axis=0).tolist(),
        "n_steps": int(all_actions_np.shape[0]),
        "action_dim": action_dim,
        "source_jsonl": str(adapter_jsonl),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Action stats saved to {out_path}  ({stats['n_steps']:,} steps)")
    return stats
