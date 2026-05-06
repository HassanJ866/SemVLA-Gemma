"""
End-to-end adapter + brain evaluation on LIBERO-Spatial tasks.

Runs the 5-step inference chain inside the LIBERO simulator for N trials per
task and reports success rate per task and mean across all tasks.

Usage:
    python -m models.adapter.eval \
        --config-name=libero_full \
        brain_ckpt=ckpts/brain_phase1/final \
        adapter_ckpt=ckpts/franka_7dof/final \
        action_stats=ckpts/franka_7dof/final/action_stats.json
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def run_eval(
    brain_ckpt: str,
    adapter_ckpt: str,
    action_stats: str,
    data_dir: str,
    n_trials: int = 10,
    max_steps: int = 300,
    n_flow_steps: int = 10,
    device: str | None = None,
    cache_graph: bool = True,
    cache_every: int = 5,
) -> dict:
    from models.middleware.chain import InferenceChain
    from envs.libero_wrapper import LiberoEnv

    chain = InferenceChain(
        brain_ckpt=brain_ckpt,
        adapter_ckpt=adapter_ckpt,
        action_stats=action_stats,
        n_flow_steps=n_flow_steps,
        device=device,
        cache_graph=cache_graph,
        cache_every=cache_every,
    )

    data_dir = Path(data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))

    results = {}
    all_successes = []

    for hdf5_path in hdf5_files:
        task_successes = []
        env = LiberoEnv(hdf5_path=str(hdf5_path))
        instruction = env.instruction

        for trial in range(n_trials):
            env.reset()
            success = False
            action_chunk = None
            chunk_ptr = 0

            for step in range(max_steps):
                if action_chunk is None or chunk_ptr >= len(action_chunk):
                    image = env.get_image()
                    proprio = env.get_proprio()
                    action_chunk = chain.step(image, instruction, proprio)
                    chunk_ptr = 0

                raw_action = action_chunk[chunk_ptr]
                chunk_ptr += 1
                obs, reward, done, info = env.step(raw_action)

                if done or info.get("success", False):
                    success = True
                    break

            task_successes.append(int(success))
            log.info(f"  {hdf5_path.stem} | trial {trial+1}/{n_trials} | "
                     f"{'SUCCESS' if success else 'FAIL'}")

        sr = sum(task_successes) / len(task_successes)
        results[hdf5_path.stem] = sr
        all_successes.extend(task_successes)
        log.info(f"  Task success rate: {sr:.2f}")

    mean_sr = sum(all_successes) / len(all_successes) if all_successes else 0.0
    results["mean"] = mean_sr

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return results


@hydra.main(config_path="../../configs/eval", config_name="libero_full", version_base=None)
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    run_eval(
        brain_ckpt=cfg.brain_ckpt,
        adapter_ckpt=cfg.adapter_ckpt,
        action_stats=cfg.action_stats,
        data_dir=cfg.data_dir,
        n_trials=cfg.n_trials,
        max_steps=cfg.max_steps,
        n_flow_steps=cfg.n_flow_steps,
        device=cfg.get("device", None),
        cache_graph=cfg.get("cache_graph", True),
        cache_every=cfg.get("cache_every", 5),
    )


if __name__ == "__main__":
    main()
