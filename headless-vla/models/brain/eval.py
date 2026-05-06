"""
Brain evaluation on a held-out JSONL split.

Metrics computed:
  - JSON validity rate per task type
  - Grounding: bbox IoU (mean and median)
  - Parsing: per-relation precision, recall, F1; aggregate F1
  - Semantic action: exact-match accuracy on (axis, direction, magnitude, gripper)

Usage:
    python -m models.brain.eval \
        --ckpt ckpts/brain_phase1/final \
        --val-jsonl data/splits/libero_train_val.jsonl \
        --image-root data/images \
        [--n-samples 500]
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── metrics ────────────────────────────────────────────────────────────────────

def bbox_iou(pred: list, gt: list) -> float:
    x1 = max(pred[0], gt[0]); y1 = max(pred[1], gt[1])
    x2 = min(pred[2], gt[2]); y2 = min(pred[3], gt[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_gt   = (gt[2]  - gt[0])  * (gt[3]  - gt[1])
    return inter / (area_pred + area_gt - inter + 1e-9)


def triplet_set(triplets: list) -> set:
    return {tuple(t) for t in triplets if len(t) == 3}


def graph_metrics(pred_triplets: list, gt_triplets: list) -> dict:
    pred_set = triplet_set(pred_triplets)
    gt_set   = triplet_set(gt_triplets)
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1}


# ── evaluation loop ────────────────────────────────────────────────────────────

def evaluate(ckpt: str, val_jsonl: str, image_root: str, n_samples: int | None):
    from models.brain.infer import BrainInference
    brain = BrainInference(ckpt)

    records = []
    with open(val_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if n_samples:
        import random
        random.shuffle(records)
        records = records[:n_samples]

    image_root = Path(image_root)
    validity = defaultdict(lambda: [0, 0])
    ious = []
    graph_f1s = []
    per_rel: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    action_correct = 0
    action_total = 0

    for rec in records:
        task = rec["task_type"]
        gt = json.loads(rec["output"])
        image = None
        img_rel = rec.get("image")
        if img_rel:
            p = image_root / img_rel
            if p.exists():
                image = Image.open(p).convert("RGB")

        validity[task][1] += 1

        try:
            if task == "grounding":
                pred = brain.ground(image, rec["instruction"])
                assert "object" in pred and "bbox" in pred and len(pred["bbox"]) == 4
                validity[task][0] += 1
                ious.append(bbox_iou(pred["bbox"], gt["bbox"]))

            elif task == "parsing":
                bboxes = rec.get("bboxes", [])
                pred = brain.parse(image, bboxes)
                assert "triplets" in pred
                validity[task][0] += 1
                m = graph_metrics(pred["triplets"], gt["triplets"])
                graph_f1s.append(m["f1"])
                for triplet_list, sign in [(pred["triplets"], 1), (gt["triplets"], -1)]:
                    for t in triplet_list:
                        if len(t) != 3:
                            continue
                        rel = t[1]
                        pair = (t[0], t[2])
                        if sign == 1:
                            per_rel[rel]["fp"] += 1
                        else:
                            per_rel[rel]["fn"] += 1
                for t in pred["triplets"]:
                    if len(t) != 3:
                        continue
                    rel = t[1]
                    pair = (t[0], t[2])
                    if any(g[0] == t[0] and g[1] == t[1] and g[2] == t[2]
                           for g in gt["triplets"]):
                        per_rel[rel]["tp"] += 1
                        per_rel[rel]["fp"] -= 1

            elif task == "action":
                pred = brain.semantic_action(
                    rec["instruction"],
                    rec.get("scene_graph", {"triplets": []}),
                    rec.get("proprio", []),
                )
                assert all(k in pred for k in ("axis", "direction", "magnitude", "gripper"))
                validity[task][0] += 1
                action_total += 1
                if (pred["axis"] == gt["axis"]
                        and pred["direction"] == gt["direction"]
                        and pred["magnitude"] == gt["magnitude"]
                        and pred["gripper"] == gt["gripper"]):
                    action_correct += 1

        except Exception as e:
            log.warning(f"Sample failed ({task}): {e}")

    report = {}
    for task, (valid, total) in validity.items():
        report[f"validity/{task}"] = valid / max(total, 1)
    report["grounding/iou_mean"]   = float(np.mean(ious))   if ious else 0.0
    report["grounding/iou_median"] = float(np.median(ious)) if ious else 0.0
    report["parsing/graph_f1"]     = float(np.mean(graph_f1s)) if graph_f1s else 0.0
    if action_total:
        report["action/exact_match"] = action_correct / action_total

    for rel, counts in per_rel.items():
        tp = max(counts["tp"], 0)
        fp = max(counts["fp"], 0)
        fn = max(counts["fn"], 0)
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        report[f"parsing/rel/{rel}/f1"] = 2 * p * r / (p + r + 1e-9)

    print("\n=== Brain Evaluation Results ===")
    for k, v in sorted(report.items()):
        print(f"  {k}: {v:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--val-jsonl", default="data/splits/libero_train_val.jsonl")
    parser.add_argument("--image-root", default="data/images")
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.ckpt, args.val_jsonl, args.image_root, args.n_samples)


if __name__ == "__main__":
    main()
