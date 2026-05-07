# Words Speak Louder Than Actions
## Headless VLA via Scene Graph Grounding and Per-Embodiment Adapters

A three-layer robotic control system:

| Layer | Component | Description |
|---|---|---|
| 1 | **Semantic Brain** | Gemma 4 fine-tuned on 3-task curriculum. Frozen after Phase 1. |
| 2 | **Embodiment Adapter** | ~3–8M flow-matching transformer per arm class. |
| 3 | **Hardware Driver** | LIBERO env / ROS2. Not trained. |

---

## Three Training Tasks

**Task 1 — Parsing:** `image + bboxes → scene graph`

Input: image + list of detected bounding boxes. Output: `{"triplets": [["bowl", "is_on_top_of", "plate"], ...]}`

**Task 2 — Grounding:** `image + task text → object bbox`

Input: image + natural language instruction. Output: `{"object": "akita_black_bowl_1", "bbox": [x1, y1, x2, y2]}`

**Task 3 — Task Synthesis:** `image + src_bbox + dst_bbox + src_local_graph → task string`

Input: image + two bounding boxes (object to move + destination) + the source object's spatial context graph. Output: `{"task": "pick up the second black bowl on the ramekin and place it on the plate"}`. Trained on synthetic data: all valid (src, dst) pairs across the 7 scene objects, with task strings generated from spatial-relation templates.

---

## Inference Chain (per control step)

```
Step 1  GROUNDING  — brain: image + instruction → target bbox
Step 2  PARSING    — brain: image + bboxes → scene graph
Step 3  SYNTHESIS  — brain: image + src_bbox + dst_bbox + local_graph → task string
Step 4  MIDDLEWARE — encode task string → task_emb; encode graph → graph_feats
Step 5  ADAPTER    — flow-matching: (task_emb, graph_feats, state, noisy_chunk, tau) → action chunk
```

---

## Data

`data/libero_spatial_v5/` — 10 LIBERO-Spatial augmented HDF5 files. 500 demos, ~62k steps.

Each file has 7 objects per scene: `akita_black_bowl_1`, `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`, `plate_1`, `wooden_cabinet_1`, `flat_stove_1`.

HDF5 already contains per-frame: `agentview_rgb`, `agentview_bboxes`, `agentview_scene_graph`, `robot_states` (9-dim), `actions` (7-dim OSC_POSE delta).

---

## Quickstart

```bash
# 1. Build 3-task JSONL from HDF5
python -m data.annotation.libero_build_graph \
    --data-dir data/libero_spatial_v5 \
    --out data/splits/libero_train.jsonl \
    --image-root data/images

# 2. Label adapter windows + compute norm stats
python -m data.annotation.motion_primitive_label \
    --data-dir data/libero_spatial_v5 \
    --out data/splits/adapter_franka.jsonl
python -c "from models.middleware.normalize import compute_and_save_stats; \
           compute_and_save_stats('data/splits/adapter_franka.jsonl', \
                                  'ckpts/franka_7dof/action_stats.json')"

# 3. Train brain (Phase 1, ~200 GPU-hours)
bash scripts/train_brain.sh

# 4. Train adapter
bash scripts/train_adapter.sh franka_7dof

# 5. Evaluate
bash scripts/eval_libero.sh
```

---

## Repository Layout

```
headless-vla/
├── configs/
│   ├── brain/phase1_libero.yaml      # LoRA fine-tune config
│   └── adapter/franka_7dof.yaml      # Adapter training config
├── data/
│   ├── annotation/
│   │   ├── libero_build_graph.py     # HDF5 → 3-task JSONL
│   │   ├── libero_extract_bboxes.py  # Standalone bbox extractor
│   │   └── motion_primitive_label.py # HDF5 → adapter windows
│   ├── schemas/                      # JSON output schemas
│   └── splits/                       # Generated JSONL splits
├── models/
│   ├── brain/
│   │   ├── prompts.py                # 3 prompt templates
│   │   ├── train.py                  # LoRA fine-tuning loop
│   │   ├── infer.py                  # Frozen inference + constrained decoding
│   │   └── eval.py                   # Brain metrics
│   ├── adapter/
│   │   ├── architecture.py           # SemanticActionAdapter
│   │   ├── flow_matching.py          # CFM loss + Euler denoiser
│   │   ├── train.py                  # Adapter training loop
│   │   └── eval.py                   # LIBERO rollout evaluation
│   └── middleware/
│       ├── chain.py                  # 5-step inference chain
│       ├── graph_encoder.py          # Bag-of-relations encoder (64-dim)
│       └── normalize.py              # Action norm/denorm
├── envs/
│   ├── libero_wrapper.py             # LIBERO sim + HDF5 replay fallback
│   └── ros2_wrapper.py               # Stub for real-arm deployment
├── tests/                            # pytest suite (32 tests, no GPU needed)
├── scripts/                          # train_brain.sh, train_adapter.sh, eval_libero.sh
└── pyproject.toml
```

---

## Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Brain model | `unsloth/gemma-4-E4B-it` | Unsloth-optimized Gemma 4 efficient 4B multimodal |
| Fine-tune method | LoRA r=32, α=64, vision layers frozen | Speed; full FT is ablation |
| Data collation | `UnslothVisionDataCollator` | Required for Gemma4 — generates `image_position_ids` correctly |
| Graph encoder | Bag-of-relations, 64-dim | Simple default; GNN upgrade if F1 transfer is poor |
| Adapter conditioning | Mean-pooled task text embedding (384-dim) | Replaces discrete semantic_ids; more expressive |
| Action normalisation | Per-dim min/max, computed before training | Must recompute if data changes |
| Constrained decoding | `outlines` JSON schema | Hard requirement; malformed JSON = hard failure |

---

## Phase Gates

| Phase | Gate |
|---|---|
| 1 Brain | JSON validity ≥ 0.97; grounding IoU ≥ 0.70; parsing F1 ≥ 0.80 |
| 2 Adapter | LIBERO-Spatial success ≥ 60%; end-to-end p50 ≤ 200ms |
