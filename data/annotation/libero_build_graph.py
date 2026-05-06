"""
Build the 3-task training JSONL from LIBERO-Spatial augmented HDF5 files.

The HDF5 files already contain:
  - agentview_rgb        (H, W, 3) uint8 per step
  - agentview_bboxes     JSON list[dict] per step
  - agentview_scene_graph JSON list[list[triplet]] per step
  - actions              (T, 7) float64
  - robot_states         (T, 9) float64
  - obs/ee_states        (T, 6) float64
  - obs/gripper_states   (T, 2) float64

Each JSONL line is one training sample of task_type in
{grounding, parsing, task_synthesis}. The three tasks are interleaved at 33/33/33.

Images are saved to <image_root>/<task_slug>/<demo>/<step>.png and the JSONL
stores the relative path.

Usage:
    python -m data.annotation.libero_build_graph \
        --data-dir data/libero_spatial_v5 \
        --out data/splits/libero_train.jsonl \
        --image-root data/images \
        [--val-frac 0.1]
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np


# ── object vocabulary ─────────────────────────────────────────────────────────

# Hardcoded (src, dst, task_string) triples for task synthesis.
# Both bowl objects map to "the black bowl" — no "second black bowl" ambiguity.
# Destinations cover all physically sensible placements for each movable object.
TASK_STRINGS = [
    ("akita_black_bowl_1", "plate_1",                        "pick up the black bowl and place it on the plate"),
    ("akita_black_bowl_1", "flat_stove_1",                   "pick up the black bowl and place it on the stove"),
    ("akita_black_bowl_1", "cookies_1",                      "pick up the black bowl and place it on the cookie box"),
    ("akita_black_bowl_1", "glazed_rim_porcelain_ramekin_1", "pick up the black bowl and place it on the ramekin"),
    ("akita_black_bowl_1", "wooden_cabinet_1",               "pick up the black bowl and place it on the wooden cabinet"),
    ("akita_black_bowl_2", "plate_1",                        "pick up the black bowl and place it on the plate"),
    ("akita_black_bowl_2", "flat_stove_1",                   "pick up the black bowl and place it on the stove"),
    ("akita_black_bowl_2", "cookies_1",                      "pick up the black bowl and place it on the cookie box"),
    ("akita_black_bowl_2", "glazed_rim_porcelain_ramekin_1", "pick up the black bowl and place it on the ramekin"),
    ("akita_black_bowl_2", "wooden_cabinet_1",               "pick up the black bowl and place it on the wooden cabinet"),
    ("cookies_1",          "plate_1",                        "pick up the cookie box and place it on the plate"),
    ("cookies_1",          "flat_stove_1",                   "pick up the cookie box and place it on the stove"),
    ("cookies_1",          "wooden_cabinet_1",               "pick up the cookie box and place it on the wooden cabinet"),
    ("glazed_rim_porcelain_ramekin_1", "plate_1",            "pick up the ramekin and place it on the plate"),
    ("glazed_rim_porcelain_ramekin_1", "flat_stove_1",       "pick up the ramekin and place it on the stove"),
]


# ── helpers ────────────────────────────────────────────────────────────────────

def _slugify(s: str) -> str:
    return s.replace(" ", "_").replace("/", "_")[:80]


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_image(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # HDF5 stores RGB; cv2 writes BGR
    cv2.imwrite(str(path), arr[:, :, ::-1])


def _load_json_field(raw) -> Any:
    if isinstance(raw, bytes):
        return json.loads(raw.decode())
    if isinstance(raw, np.ndarray) and raw.dtype.kind in ("S", "U", "O"):
        return json.loads(raw.item().decode() if isinstance(raw.item(), bytes) else raw.item())
    return raw


def _pick_target_object(instruction: str, bboxes: dict) -> tuple[str, list[int]] | None:
    """Heuristic: find the object name that most closely matches instruction tokens."""
    instr_lower = instruction.lower()
    best_name = None
    best_score = 0
    for obj_name, bbox in bboxes.items():
        words = obj_name.replace("_", " ").replace("1", "").replace("2", "").strip().lower().split()
        score = sum(1 for w in words if w in instr_lower and len(w) > 2)
        if score > best_score:
            best_score = score
            best_name = obj_name
    if best_name is None:
        best_name = next(iter(bboxes))
    return best_name, bboxes[best_name]


def _generate_task_synthesis_record(
    src_name: str, dst_name: str, task_str: str,
    bboxes_step: dict, sg_triplets: list, img_rel: str,
) -> dict | None:
    """Build one task_synthesis JSONL record for a hardcoded (src, dst, task_str) triple."""
    if src_name not in bboxes_step or dst_name not in bboxes_step:
        return None

    src_graph = [t for t in sg_triplets if len(t) == 3 and t[0] == src_name]

    return {
        "task_type": "task_synthesis",
        "image": img_rel,
        "src_name": src_name,
        "src_bbox": bboxes_step[src_name],
        "dst_name": dst_name,
        "dst_bbox": bboxes_step[dst_name],
        "src_graph": src_graph,
        "output": json.dumps({"task": task_str}),
    }


# ── per-demo processing ────────────────────────────────────────────────────────

def process_demo(demo_key: str, demo: h5py.Group, instruction: str,
                 image_root: Path, task_slug: str,
                 subsample: int = 1) -> list[dict]:
    """Return list of JSONL records (all 3 task types) for one demo."""
    rgb = demo["obs"]["agentview_rgb"][()]             # (T, H, W, 3)
    robot_states = demo["robot_states"][()]             # (T, 9)

    bboxes_all = _load_json_field(demo["obs"]["agentview_bboxes"][()])
    sg_all = _load_json_field(demo["obs"]["agentview_scene_graph"][()])

    T = len(rgb)

    records = []
    for t in range(0, T, subsample):
        bboxes_step: dict = bboxes_all[t] if t < len(bboxes_all) else {}
        sg_triplets: list = sg_all[t] if t < len(sg_all) else []

        # ── save image ──────────────────────────────────────────────────────
        img_rel = f"{task_slug}/{demo_key}/step_{t:04d}.png"
        img_path = image_root / img_rel
        if not img_path.exists():
            _save_image(rgb[t], img_path)

        # ── Task 1: grounding ───────────────────────────────────────────────
        if bboxes_step:
            obj_name, bbox = _pick_target_object(instruction, bboxes_step)
            records.append({
                "task_type": "grounding",
                "image": img_rel,
                "instruction": instruction,
                "output": json.dumps({"object": obj_name, "bbox": bbox}),
            })

        # ── Task 2: parsing ─────────────────────────────────────────────────
        if sg_triplets:
            bboxes_list = [{"name": k, "bbox": v} for k, v in bboxes_step.items()]
            records.append({
                "task_type": "parsing",
                "image": img_rel,
                "bboxes": bboxes_list,
                "output": json.dumps({"triplets": sg_triplets}),
            })

        # ── Task 3: task synthesis (1 random hardcoded triple per frame) ────
        if bboxes_step:
            candidates = [
                (src, dst, task_str)
                for src, dst, task_str in TASK_STRINGS
                if src in bboxes_step and dst in bboxes_step
            ]
            if candidates:
                src_name, dst_name, task_str = random.choice(candidates)
                rec = _generate_task_synthesis_record(
                    src_name, dst_name, task_str, bboxes_step, sg_triplets, img_rel
                )
                if rec is not None:
                    records.append(rec)

    return records


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.expanduser("~/vlm_benchmarking/data/libero_spatial_v5"))
    parser.add_argument("--out", default="data/splits/libero_train.jsonl")
    parser.add_argument("--image-root", default="data/images")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="Fraction of demos held out as validation set")
    parser.add_argument("--subsample", type=int, default=1,
                        help="Step stride for frame subsampling (1=every frame)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    val_path = out_path.with_stem(out_path.stem + "_val")
    image_root = Path(args.image_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files in {data_dir}")

    train_records: list[dict] = []
    val_records: list[dict] = []
    manifest = []

    for hdf5_path in hdf5_files:
        print(f"Processing {hdf5_path.name} …")
        with h5py.File(hdf5_path, "r") as f:
            info = json.loads(f["data"].attrs["problem_info"])
            instruction = info["language_instruction"]
            task_slug = _slugify(instruction)
            demo_keys = sorted(f["data"].keys())

            n_val = max(1, int(len(demo_keys) * args.val_frac))
            random.shuffle(demo_keys)
            val_keys = set(demo_keys[:n_val])
            train_keys = demo_keys[n_val:]

            file_records = 0
            for demo_key in demo_keys:
                recs = process_demo(
                    demo_key, f["data"][demo_key], instruction,
                    image_root, task_slug, args.subsample,
                )
                if demo_key in val_keys:
                    val_records.extend(recs)
                else:
                    train_records.extend(recs)
                file_records += len(recs)

            manifest.append({
                "file": hdf5_path.name,
                "md5": _file_hash(str(hdf5_path)),
                "task": instruction,
                "demos": len(demo_keys),
                "val_demos": n_val,
                "records": file_records,
            })

    # shuffle and write
    random.shuffle(train_records)
    random.shuffle(val_records)

    with open(out_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    with open(val_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump({
            "train_records": len(train_records),
            "val_records": len(val_records),
            "sources": manifest,
        }, mf, indent=2)

    print(f"\nTrain records: {len(train_records):,}")
    print(f"Val   records: {len(val_records):,}")
    print(f"Manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
