"""
ROS2 wrapper stub for real-arm deployment. Out of scope for Phase 1/2 training.
Implement when deploying to a physical arm.
"""

import numpy as np
from PIL import Image


class ROS2Env:
    """Stub: replace with real ROS2 node implementation for physical arm."""

    def __init__(self, instruction: str, action_topic: str = "/arm/cmd_vel",
                 image_topic: str = "/camera/color/image_raw",
                 state_topic: str = "/arm/joint_states"):
        self.instruction   = instruction
        self.action_topic  = action_topic
        self.image_topic   = image_topic
        self.state_topic   = state_topic
        raise NotImplementedError(
            "ROS2Env is a stub. Implement ROS2 subscriptions/publishers for real-arm use."
        )

    def reset(self) -> dict:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        raise NotImplementedError

    def get_image(self) -> Image.Image:
        raise NotImplementedError

    def get_proprio(self) -> np.ndarray:
        raise NotImplementedError
