#!/usr/bin/env bash
# Latency benchmark: measures per-step inference time for all 5 chain stages.
# Acceptance gate: end-to-end p50 <= 200ms (>= 5 Hz).
# Run from the headless-vla/ root directory.
set -euo pipefail

BRAIN_CKPT=${1:-ckpts/brain_phase1/final}
ADAPTER_CKPT=${2:-ckpts/franka_7dof/final}
ACTION_STATS=${3:-ckpts/franka_7dof/final/action_stats.json}
DEVICE=${4:-cuda}
N_WARMUP=5
N_ITERS=50

python - <<'EOF'
import sys, time, json, statistics
import numpy as np
import torch
from PIL import Image

brain_ckpt   = sys.argv[1] if len(sys.argv) > 1 else "ckpts/brain_phase1/final"
adapter_ckpt = sys.argv[2] if len(sys.argv) > 2 else "ckpts/franka_7dof/final"
action_stats = sys.argv[3] if len(sys.argv) > 3 else "ckpts/franka_7dof/final/action_stats.json"
device       = sys.argv[4] if len(sys.argv) > 4 else "cuda"

from models.brain.infer import BrainInference
from models.adapter.architecture import SemanticActionAdapter
from models.adapter.flow_matching import flow_matching_inference
from models.middleware.enums import semantic_action_to_ids
from models.middleware.graph_encoder import encode_graph_tensor
from models.middleware.normalize import ActionNormalizer

brain   = BrainInference(brain_ckpt, device=device)
norm    = ActionNormalizer(action_stats)
with open(f"{adapter_ckpt}/adapter_config.json") as f:
    cfg = json.load(f)
adapter = SemanticActionAdapter(**cfg).to(device)
adapter.load_state_dict(torch.load(f"{adapter_ckpt}/adapter.pt",
                                   map_location=device, weights_only=True))
adapter.eval()

dummy_image  = Image.new("RGB", (128, 128), (120, 100, 80))
instruction  = "pick up the black bowl and place it on the plate"
proprio      = np.zeros(9, dtype=np.float32)
sem_ids_t    = torch.tensor([[0, 3, 5, 8]], dtype=torch.long, device=device)
graph_t      = encode_graph_tensor([], device=device)
state_t      = torch.zeros(1, 9, device=device)

n_warmup = 5
n_iters  = 50

# warmup
for _ in range(n_warmup):
    brain.ground(dummy_image, instruction)

times = {"grounding": [], "parsing": [], "action": [], "adapter": [], "total": []}
for _ in range(n_iters):
    t0 = time.perf_counter()
    grounding = brain.ground(dummy_image, instruction)
    t1 = time.perf_counter()
    scene_graph = brain.parse(dummy_image, [])
    t2 = time.perf_counter()
    sem_action = brain.semantic_action(instruction, scene_graph, proprio.tolist())
    t3 = time.perf_counter()
    sem_ids_t = torch.tensor([semantic_action_to_ids(sem_action)],
                              dtype=torch.long, device=device)
    with torch.no_grad():
        chunk = flow_matching_inference(adapter, sem_ids_t, graph_t, state_t,
                                        cfg["chunk_size"], cfg["action_dim"],
                                        n_steps=10, device=device)
    t4 = time.perf_counter()

    times["grounding"].append((t1-t0)*1000)
    times["parsing"].append((t2-t1)*1000)
    times["action"].append((t3-t2)*1000)
    times["adapter"].append((t4-t3)*1000)
    times["total"].append((t4-t0)*1000)

print("\n=== Latency Benchmark (ms) ===")
for stage, vals in times.items():
    p50  = statistics.median(vals)
    p95  = sorted(vals)[int(0.95*len(vals))]
    mean = statistics.mean(vals)
    print(f"  {stage:<12} p50={p50:.1f}ms  p95={p95:.1f}ms  mean={mean:.1f}ms")

total_p50 = statistics.median(times["total"])
hz = 1000.0 / total_p50
print(f"\n  End-to-end p50: {total_p50:.1f}ms  ({hz:.1f} Hz)")
gate = "PASS" if total_p50 <= 200 else "FAIL (> 200ms target)"
print(f"  Latency gate: {gate}")
EOF
