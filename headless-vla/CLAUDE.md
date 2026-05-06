# CLAUDE.md — headless-vla

## Project in one sentence
Fine-tune Gemma 4 on a 3-task curriculum (parsing, grounding, task-synthesis) then train a small flow-matching adapter per robot arm on LIBERO-Spatial augmented data.

## Repo layout (critical paths only)
```
data/libero_spatial_v5/     10 HDF5 files, 500 demos, ~62k steps, 7 objects per scene
data/annotation/
  libero_build_graph.py     HDF5 → 3-task JSONL (brain training data)
  motion_primitive_label.py HDF5 → adapter windows JSONL
data/splits/                Generated outputs (not committed)
data/schemas/               JSON output schemas for constrained decoding
models/brain/
  prompts.py                3 prompt templates + format_training_sample()
  train.py                  LoRA fine-tune loop (Hydra config)
  infer.py                  BrainInference class (outlines constrained decoding)
  eval.py                   Brain metrics (IoU, F1, exact-match)
models/adapter/
  architecture.py           SemanticActionAdapter (~3-8M params)
  flow_matching.py          CFM loss + Euler denoiser
  train.py                  Adapter training loop (Hydra config)
  eval.py                   LIBERO rollout evaluation
models/middleware/
  chain.py                  5-step inference chain (the runtime loop)
  graph_encoder.py          Bag-of-relations encoder → 64-dim float vector
  normalize.py              ActionNormalizer + compute_and_save_stats()
envs/libero_wrapper.py      LIBERO sim wrapper; falls back to HDF5 replay if LIBERO not installed
configs/brain/phase1_libero.yaml
configs/adapter/franka_7dof.yaml
configs/eval/libero_full.yaml
```

## The three tasks

**Task 1 — parsing** (`task_type: "parsing"`)
- Input: `<image>` + bboxes list `[{"name":…,"bbox":[x1,y1,x2,y2]},…]`
- Output: `{"triplets": [["bowl","is_on_top_of","plate"],…]}`

**Task 2 — grounding** (`task_type: "grounding"`)
- Input: `<image>` + instruction string
- Output: `{"object": "akita_black_bowl_1", "bbox": [x1,y1,x2,y2]}`

**Task 3 — task synthesis** (`task_type: "task_synthesis"`)
- Input: `<image>` + src_bbox + dst_bbox + source object's local graph triplets
- Output: `{"task": "pick up the second black bowl on the ramekin and place it on the plate"}`
- Trained on **synthetic data**: 1 random valid (src,dst) pair per frame; task strings generated from spatial-relation templates using OBJ_NAMES map
- Replaces the old semantic-action task (axis/direction/magnitude/gripper) which caused model to overfit to bowl→plate

## Data facts
- All 10 scenes share the same 7 objects: `akita_black_bowl_1`, `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`, `plate_1`, `wooden_cabinet_1`, `flat_stove_1`
- HDF5 relations used: `is_on_top_of`, `is_below_of`, `is_left_of`, `is_right_of`, `is_in_front_of`, `is_behind`, `is_inside`, `contains`
- HDF5 relation names differ from `relations_v1.json` schema (`is_on_top_of` ≠ `is_on`, `is_below_of` ≠ `is_under`) — use HDF5 strings as-is
- `robot_states` shape: (T, 9); `actions` shape: (T, 7) OSC_POSE delta

## Adapter input (post-Task-3 redesign)
- `task_proj: Linear(384, hidden)` — mean-pooled Gemma token embeddings of the synthesized task string
- NOT `semantic_embed: Embedding(20, hidden)` (that was the old axis/dir/mag/grip design)
- `encode_task_text(text)` helper in `models/middleware/chain.py` handles encoding at inference time
- Adapter JSONL records have `task_emb` (384-dim float list) + `task_text` fields

## Mandatory before adapter training
```python
from models.middleware.normalize import compute_and_save_stats
compute_and_save_stats("data/splits/adapter_franka.jsonl",
                       "ckpts/franka_7dof/action_stats.json", action_dim=7)
```
Stats must be recomputed whenever the data changes.

## Run commands
```bash
# Build brain JSONL
python -m data.annotation.libero_build_graph --data-dir data/libero_spatial_v5 --out data/splits/libero_train.jsonl --image-root data/images

# Build adapter JSONL
python -m data.annotation.motion_primitive_label --data-dir data/libero_spatial_v5 --out data/splits/adapter_franka.jsonl

# Train brain
python -m models.brain.train --config-name=phase1_libero

# Train adapter
python -m models.adapter.train --config-name=franka_7dof

# Evaluate
python -m models.adapter.eval --config-name=libero_full brain_ckpt=ckpts/brain_phase1/final adapter_ckpt=ckpts/franka_7dof/final

# Tests (no GPU needed, ~3s)
python -m pytest tests/ -v
```

## Key decisions (do not change without updating this file)
| Item | Value | Reason |
|---|---|---|
| Brain model ID | `google/gemma-3-4b-it` | Closest available Gemma 4 multimodal; verify HF at run time |
| LoRA | r=32, α=64, target q/k/v/o_proj | Default; full FT is ablation only |
| Graph encoder | Bag-of-relations, 64-dim, L2-normed | Simple default; GNN upgrade deferred |
| Task text encoder | Mean-pool Gemma token embeddings, 384-dim | No extra model; fast at inference |
| Action normalisation | Per-dim min/max from training data | Most common adapter failure mode if skipped |
| Constrained decoding | `outlines` JSON schema | Malformed JSON = hard failure, not warning |
| Chunk size | 16 timesteps | LIBERO default; tunable per arm |
| Flow steps | 10 Euler steps | Fixed for inference |
| Tau distribution | Beta(1.5, 1.0) | Biases toward near-clean samples |

## Phase gates
- Brain: JSON validity ≥ 0.97; grounding IoU ≥ 0.70; parsing F1 ≥ 0.80; task-synthesis output must be valid JSON
- Adapter: LIBERO-Spatial success ≥ 60%; end-to-end p50 ≤ 200ms on Jetson Orin
