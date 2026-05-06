"""
Extract per-frame bounding boxes from LIBERO-Spatial HDF5 files.

The augmented HDF5 files already contain `agentview_bboxes` as a JSON-encoded
list of per-step dicts: [{obj_name: [x1,y1,x2,y2], ...}, ...].
This script normalises that to a flat list of JSONL records keyed by
(task_file, demo_id, step_idx) for downstream use.

Usage:
    python -m data.annotation.libero_extract_bboxes \
        --data-dir data/libero_spatial_v5 \
        --out data/splits/libero_bboxes.jsonl
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py


def _file_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_bboxes_from_hdf5(hdf5_path: Path) -> list[dict]:
    records = []
    with h5py.File(hdf5_path, "r") as f:
        info = json.loads(f["data"].attrs["problem_info"])
        instruction = info["language_instruction"]
        demo_keys = sorted(f["data"].keys())
        for demo_key in demo_keys:
            demo = f["data"][demo_key]
            bboxes_raw = demo["obs"]["agentview_bboxes"][()]
            if isinstance(bboxes_raw, bytes):
                bboxes_per_step = json.loads(bboxes_raw.decode())
            else:
                bboxes_per_step = bboxes_raw.tolist()

            for step_idx, bbox_dict in enumerate(bboxes_per_step):
                records.append({
                    "source_file": hdf5_path.name,
                    "demo_id": demo_key,
                    "step_idx": step_idx,
                    "instruction": instruction,
                    "bboxes": bbox_dict,
                })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.expanduser("~/vlm_benchmarking/data/libero_spatial_v5"))
    parser.add_argument("--out", default="data/splits/libero_bboxes.jsonl")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

    total = 0
    manifest = []
    with open(out_path, "w") as out_f:
        for hdf5_path in hdf5_files:
            records = extract_bboxes_from_hdf5(hdf5_path)
            for rec in records:
                out_f.write(json.dumps(rec) + "\n")
            total += len(records)
            manifest.append({
                "file": hdf5_path.name,
                "md5": _file_hash(str(hdf5_path)),
                "records": len(records),
            })
            print(f"  {hdf5_path.name}: {len(records)} step records")

    manifest_path = out_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as mf:
        json.dump({"total_records": total, "sources": manifest}, mf, indent=2)

    print(f"\nWrote {total} bbox records to {out_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
