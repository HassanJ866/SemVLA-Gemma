"""
LIBERO environment wrapper for the adapter evaluation loop.

Wraps a single LIBERO task (one HDF5 file) and exposes a gym-like interface:
  - reset()        → initial obs dict
  - step(action)   → (obs, reward, done, info)
  - get_image()    → PIL.Image (agentview)
  - get_proprio()  → np.ndarray robot state

The wrapper works in two modes:
  1. Simulator mode: uses the LIBERO Python package to launch the MuJoCo env.
  2. Replay mode (no LIBERO install): cycles through demo frames from the HDF5
     for testing purposes. Activated when LIBERO is not importable.
"""

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

try:
    import libero.libero.envs  # noqa: F401
    _LIBERO_AVAILABLE = True
except ImportError:
    _LIBERO_AVAILABLE = False
    log.warning("LIBERO package not found; falling back to HDF5 replay mode.")


class LiberoEnv:
    """
    Parameters
    ----------
    hdf5_path   Path to the LIBERO HDF5 demo file for this task.
    camera_name Camera used for observations (default: agentview).
    """

    def __init__(self, hdf5_path: str, camera_name: str = "agentview"):
        self.hdf5_path   = Path(hdf5_path)
        self.camera_name = camera_name
        self._env        = None
        self._replay_demo = None
        self._replay_step = 0

        with h5py.File(self.hdf5_path, "r") as f:
            info = json.loads(f["data"].attrs["problem_info"])
            self.instruction = info["language_instruction"]
            self._env_args = json.loads(f["data"].attrs["env_args"])
            self._demo_keys = sorted(f["data"].keys())

        if _LIBERO_AVAILABLE:
            self._init_simulator()
        else:
            log.info(f"Replay mode: {self.hdf5_path.name}")

    # ── simulator mode ────────────────────────────────────────────────────

    def _init_simulator(self):
        from libero.libero.envs import OffScreenRenderEnv
        env_kwargs = self._env_args.get("env_kwargs", {})
        env_kwargs.setdefault("has_renderer", False)
        env_kwargs.setdefault("has_offscreen_renderer", True)
        env_kwargs.setdefault("use_camera_obs", True)
        env_kwargs.setdefault("camera_names", ["agentview", "robot0_eye_in_hand"])
        env_kwargs.setdefault("camera_heights", 128)
        env_kwargs.setdefault("camera_widths", 128)

        bddl_file = self._env_args.get("bddl_file",
                                        self._env_args.get("bddl_file_name", ""))
        self._env = OffScreenRenderEnv(**env_kwargs, bddl_file=bddl_file)

    def reset(self) -> dict:
        if self._env is not None:
            obs = self._env.reset()
            self._obs = obs
            return obs
        else:
            return self._replay_reset()

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        if self._env is not None:
            obs, reward, done, info = self._env.step(action)
            self._obs = obs
            return obs, float(reward), bool(done), info
        else:
            return self._replay_step_fn(action)

    def get_image(self) -> Image.Image:
        if self._env is not None:
            key = f"{self.camera_name}_image"
            img_arr = self._obs[key]  # (H, W, 3) uint8
            return Image.fromarray(img_arr)
        else:
            return self._replay_image()

    def get_proprio(self) -> np.ndarray:
        if self._env is not None:
            return self._obs.get("robot0_proprio-state",
                                  np.zeros(9, dtype=np.float32))
        else:
            return self._replay_proprio()

    def close(self):
        if self._env is not None:
            self._env.close()

    # ── replay mode (no LIBERO install) ──────────────────────────────────

    def _replay_reset(self) -> dict:
        import random
        demo_key = random.choice(self._demo_keys)
        with h5py.File(self.hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            self._replay_rgb   = demo["obs"][f"{self.camera_name}_rgb"][()]
            self._replay_states = demo["robot_states"][()]
            self._replay_actions = demo["actions"][()]
            self._replay_rewards = demo["rewards"][()]
            self._replay_dones  = demo["dones"][()]
        self._replay_step = 0
        return {"replay": True, "step": 0}

    def _replay_step_fn(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        t = self._replay_step
        T = len(self._replay_rgb)
        reward = float(self._replay_rewards[min(t, T-1)])
        done   = bool(self._replay_dones[min(t, T-1)]) or t >= T - 1
        self._replay_step = min(t + 1, T - 1)
        return {"replay": True, "step": t}, reward, done, {"success": done and reward > 0}

    def _replay_image(self) -> Image.Image:
        t = min(self._replay_step, len(self._replay_rgb) - 1)
        return Image.fromarray(self._replay_rgb[t])

    def _replay_proprio(self) -> np.ndarray:
        t = min(self._replay_step, len(self._replay_states) - 1)
        return self._replay_states[t].astype(np.float32)
