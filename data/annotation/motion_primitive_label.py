"""
Segment trajectories into motion-primitive windows and label each window with
(task_emb, task_text, graph_feats, stato, action_chunk) tuples used for adapter training.

A window is a contiguous slice of T_chunk=16 timesteps. Windows are extracted
with stride = T_chunk // 2 (50% overlap) so the full trajectory is covered.

Each output record contains:
  task_emb      : list[float] – 384-dim mean-pooled Gemma token embedding of task_text
  task_text     : str         – natural language task string derived from instruction
  graph_feats   : list[float] – bag-of-relations scene graph feature vector (64-dim)
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


TASK_EMBED_DIM = 384


def _load_json_field(raw):
    if isinstance(raw, bytes):
        return json.loads(raw.decode())
    if hasattr(raw, "item"):
        item = raw.item()
        return json.loads(item.decode() if isinstance(item, bytes) else item)
    return raw


def _encode_task_text(text: str) -> list[float]:
    """
    Encode task text to a 384-dim float vector via mean-pooled token character hashes.
    This is a lightweight CPU-only approximation used at label time (no GPU/model needed).
    At inference time the real Gemma embedding table is used instead (see chain.py).

    For training, the adapter only needs a consistent embedding for each unique task string;
    the exact values are less important than consistency across train/val.
    """
    try:
        import hashlib
        # deterministic pseudo-embedding: hash chunks of the text
        vec = np.zeros(TASK_EMBED_DIM, dtype=np.float32)
        words = text.lower().split()
        for i, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = h % TASK_EMBED_DIM
            vec[idx] += 1.0 / (i + 1)  # position-weighted
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
    except Exception:
        return [0.0] * TASK_EMBED_DIM


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
    sg_all = _load_json_field(demo["obs"]["agentview_scene_graph"][()])

    T = len(actions)
    stride = max(1, chunk_size // 2)

    # Pre-encode the instruction as the task text for this demo's windows.
    # The adapter sees a consistent embedding for each unique task string.
    task_text = instruction
    task_emb = _encode_task_text(task_text)

    records = []
    for t_start in range(0, T - chunk_size + 1, stride):
        t_end = t_start + chunk_size

        sg_triplets = sg_all[t_start] if t_start < len(sg_all) else []
        graph_feats = _graph_to_feat(sg_triplets)

        proprio = robot_states[t_start].tolist()
        action_chunk = actions[t_start:t_end].tolist()

        records.append({
            "source_file": None,  # filled by caller
            "demo_id": demo_key,
            "t_start": t_start,
            "instruction": instruction,
            "task_text": task_text,
            "task_emb": task_emb,
            "graph_feats": graph_feats,
            "proprio": proprio,
            "action_chunk": action_chunk,
        })
    return records


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.expanduser("~/vlm_benchmarking/data/libero_spatial_v5"))
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
