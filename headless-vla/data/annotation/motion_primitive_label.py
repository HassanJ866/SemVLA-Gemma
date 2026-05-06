"""
Segment trajectories into motion-primitive windows and label each window with
(semantic_action, scene_graph, state, action_chunk) tuples used for adapter training.

A window is a contiguous slice of T_chunk=16 timesteps. Windows are extracted
with stride = T_chunk // 2 (50% overlap) so the full trajectory is covered.

Each output record contains:
  semantic_ids  : list[int]  – [axis_id, dir_id, mag_id, gripper_id]
  graph_feats   : list[float] – one-hot + node embedding sum vector
  proprio       : list[float] – robot_states[t_start]
  action_chunk  : list[list[float]] – actions[t_start : t_start+chunk_size]
  (plus metadata for traceability)

Usage:
    python -m data.annotation.motion_primitive_label \
        --data-dir data/libero_spatial_v5 \
        --out data/splits/adapter_franka.jsonl \
        --chunk-size 16
"""

import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np


# ── enum maps (must match middleware/enums.py) ─────────────────────────────────

AXIS_TO_ID   = {"X": 0, "Y": 1, "Z": 2}
DIR_TO_ID    = {"positive": 0, "negative": 1}
MAG_TO_ID    = {"small": 0, "medium": 1, "large": 2}
GRIP_TO_ID   = {"open": 0, "close": 1, "keep": 2}


def _load_json_field(raw):
    if isinstance(raw, bytes):
        return json.loads(raw.decode())
    if hasattr(raw, "item"):
        item = raw.item()
        return json.loads(item.decode() if isinstance(item, bytes) else item)
    return raw


def _compute_mag_thresholds(actions: np.ndarray) -> dict:
    abs_deltas = np.abs(actions[:, :3]).max(axis=1)
    abs_deltas = abs_deltas[abs_deltas > 1e-6]
    if len(abs_deltas) == 0:
        return {"small": 0.01, "large": 0.04}
    return {
        "small": float(np.percentile(abs_deltas, 25)),
        "large": float(np.percentile(abs_deltas, 75)),
    }


def _label_semantic_action(actions: np.ndarray, gripper_states: np.ndarray,
                            t: int, mag_thresh: dict) -> dict:
    delta = actions[t, :3]
    abs_delta = np.abs(delta)
    ax_idx = int(np.argmax(abs_delta))
    axis = ["X", "Y", "Z"][ax_idx]
    direction = "positive" if delta[ax_idx] >= 0 else "negative"
    mag_val = abs_delta[ax_idx]
    lo, hi = mag_thresh["small"], mag_thresh["large"]
    magnitude = "small" if mag_val <= lo else ("medium" if mag_val <= hi else "large")

    if t == 0:
        gripper = "keep"
    else:
        prev = gripper_states[t - 1].mean()
        curr = gripper_states[t].mean()
        d = curr - prev
        gripper = "open" if d > 0.005 else ("close" if d < -0.005 else "keep")

    return {"axis": axis, "direction": direction, "magnitude": magnitude, "gripper": gripper}


def _semantic_ids(sem: dict) -> list[int]:
    return [
        AXIS_TO_ID[sem["axis"]],
        DIR_TO_ID[sem["direction"]],
        MAG_TO_ID[sem["magnitude"]],
        GRIP_TO_ID[sem["gripper"]],
    ]


# ── graph features ─────────────────────────────────────────────────────────────

RELATIONS = [
    "is_left_of", "is_right_of", "is_above", "is_below",
    "is_in_front_of", "is_behind", "is_on", "is_under", "is_inside", "contains",
]
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
N_RELS = len(RELATIONS)


def _graph_to_feat(triplets: list, n_dim: int = 64) -> list[float]:
    """
    Simple bag-of-relations encoding: sum of one-hot relation vectors over all
    triplets, zero-padded / truncated to n_dim.
    """
    vec = np.zeros(n_dim, dtype=np.float32)
    for triplet in triplets:
        if len(triplet) != 3:
            continue
        rel = triplet[1]
        rid = REL_TO_ID.get(rel, -1)
        if rid >= 0 and rid < n_dim:
            vec[rid] += 1.0
    # l2-normalise
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ── per-demo processing ────────────────────────────────────────────────────────

def process_demo(demo_key: str, demo: h5py.Group, instruction: str,
                 chunk_size: int) -> list[dict]:
    actions = demo["actions"][()]           # (T, 7)
    robot_states = demo["robot_states"][()] # (T, 9)
    gripper_states = demo["obs"]["gripper_states"][()] # (T, 2)
    sg_all = _load_json_field(demo["obs"]["agentview_scene_graph"][()])

    T = len(actions)
    mag_thresh = _compute_mag_thresholds(actions)
    stride = max(1, chunk_size // 2)

    records = []
    for t_start in range(0, T - chunk_size + 1, stride):
        t_end = t_start + chunk_size

        # dominant semantic label from the first step of the window
        sem = _label_semantic_action(actions, gripper_states, t_start, mag_thresh)
        ids = _semantic_ids(sem)

        sg_triplets = sg_all[t_start] if t_start < len(sg_all) else []
        graph_feats = _graph_to_feat(sg_triplets)

        proprio = robot_states[t_start].tolist()
        action_chunk = actions[t_start:t_end].tolist()

        records.append({
            "source_file": None,  # filled by caller
            "demo_id": demo_key,
            "t_start": t_start,
            "instruction": instruction,
            "semantic_ids": ids,
            "semantic_label": sem,
            "graph_feats": graph_feats,
            "proprio": proprio,
            "action_chunk": action_chunk,
        })
    return records


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/libero_spatial_v5")
    parser.add_argument("--out", default="data/splits/adapter_franka.jsonl")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    val_path = out_path.with_stem(out_path.stem + "_val")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {data_dir}")

    train_records, val_records = [], []
    for hdf5_path in hdf5_files:
        print(f"Processing {hdf5_path.name} …")
        with h5py.File(hdf5_path, "r") as f:
            info = json.loads(f["data"].attrs["problem_info"])
            instruction = info["language_instruction"]
            demo_keys = sorted(f["data"].keys())

            n_val = max(1, int(len(demo_keys) * args.val_frac))
            random.shuffle(demo_keys)
            val_set = set(demo_keys[:n_val])

            for demo_key in demo_keys:
                recs = process_demo(demo_key, f["data"][demo_key],
                                    instruction, args.chunk_size)
                for r in recs:
                    r["source_file"] = hdf5_path.name
                if demo_key in val_set:
                    val_records.extend(recs)
                else:
                    train_records.extend(recs)

    random.shuffle(train_records)
    random.shuffle(val_records)

    with open(out_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")
    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print(f"Train adapter windows: {len(train_records):,}")
    print(f"Val   adapter windows: {len(val_records):,}")
    print(f"Written to {out_path} and {val_path}")


if __name__ == "__main__":
    main()
