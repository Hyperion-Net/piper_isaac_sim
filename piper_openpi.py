"""OpenPI integration for the Piper color sorting task.

This module provides:
  - PiperSortingOpenPIEnv: openpi-compatible Environment subclass
  - PiperInputs / PiperOutputs: data transforms for the openpi policy
  - Config helpers for serving/fine-tuning

Architecture:
  - Policy server runs on a GPU machine (Ubuntu + NVIDIA GPU)
  - This client runs on any machine (macOS OK), connects via websocket
  - Observations: 8-dim joint state + 224x224 overhead RGB + language prompt
  - Actions: 9-dim (6 arm joints + 2 gripper joints + grasp trigger)

Usage with openpi server:
  1. On GPU machine: python -m openpi.serving.serve_policy --config piper_sorting
  2. On this machine: python sort_blocks.py --host <gpu_machine_ip>
"""

import dataclasses
import numpy as np
from typing import Any

from sorting_env import (
    PiperSortingEnv,
    BLOCK_NAMES,
    BLOCK_COLORS,
    ZONE_POSITIONS,
    SORT_TOLERANCE,
)

# Action dimension: 6 arm + 2 gripper + 1 grasp trigger
ACTION_DIM = 9
STATE_DIM = 8  # 6 arm joints + 2 gripper joints
IMG_SIZE = 224


@dataclasses.dataclass
class PiperInputs:
    """Transform Piper observations to openpi model input format.

    Follows the pattern from openpi's LiberoInputs / AlohaInputs.
    Maps our env observations to the dict format expected by the policy network.
    """

    action_dim: int = ACTION_DIM
    state_dim: int = STATE_DIM

    def __call__(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Convert env observation to policy input.

        Args:
            obs: dict from PiperSortingEnv.get_observation() with added "prompt" key

        Returns:
            dict with "state", "image", "prompt" keys matching openpi format
        """
        state = np.array(obs["state"], dtype=np.float32)
        assert state.shape == (self.state_dim,), f"Expected state dim {self.state_dim}, got {state.shape}"

        # Resize image to 224x224 if needed
        img = obs["images"]["overhead_0_rgb"]
        if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
            # Simple resize via nearest neighbor (avoid cv2 dependency)
            h, w = img.shape[:2]
            y_idx = (np.arange(IMG_SIZE) * h / IMG_SIZE).astype(int)
            x_idx = (np.arange(IMG_SIZE) * w / IMG_SIZE).astype(int)
            img = img[np.ix_(y_idx, x_idx)]

        return {
            "state": state,
            "image": {
                "base_0_rgb": np.ascontiguousarray(img, dtype=np.uint8),
            },
            "prompt": obs.get("prompt", "sort the colored blocks into their matching zones"),
        }


@dataclasses.dataclass
class PiperOutputs:
    """Transform openpi model output to Piper action format.

    Maps the raw action array from the policy to the format expected by
    PiperSortingEnv.step().
    """

    action_dim: int = ACTION_DIM

    def __call__(self, raw_actions: np.ndarray) -> np.ndarray:
        """Convert policy output to env action.

        Args:
            raw_actions: array from policy, shape (chunk_size, action_dim) or (action_dim,)

        Returns:
            np.array of shape (action_dim,) — single action step
        """
        if raw_actions.ndim == 2:
            # Action chunk — take first action
            action = raw_actions[0]
        else:
            action = raw_actions

        action = np.array(action, dtype=np.float32)

        # Clip arm joints to safe ranges
        # joints 0-5: arm, 6-7: gripper, 8: grasp trigger
        action[6] = np.clip(action[6], 0.0, 0.035)
        action[7] = np.clip(action[7], -0.035, 0.0)
        if len(action) > 8:
            action[8] = np.clip(action[8], -1.0, 1.0)

        return action


class PiperSortingOpenPIEnv:
    """OpenPI-compatible environment wrapper for color sorting.

    Implements the interface expected by openpi's Runtime:
      - reset() -> observation dict
      - get_observation() -> observation dict
      - apply_action(action) -> None
      - is_episode_complete() -> bool

    This follows the pattern from examples/aloha_sim/env.py.
    """

    def __init__(self, render_size=(IMG_SIZE, IMG_SIZE), prompt=None):
        self.env = PiperSortingEnv(render_size=render_size)
        self.prompt = prompt or "sort the colored blocks into their matching zones"
        self._done = False
        self._last_info = {}
        self._inputs = PiperInputs()
        self._outputs = PiperOutputs()

    def reset(self) -> dict[str, Any]:
        obs = self.env.reset(randomize=True)
        self._done = False
        self._last_info = {}
        obs["prompt"] = self.prompt
        return self._inputs(obs)

    def get_observation(self) -> dict[str, Any]:
        obs = self.env.get_observation()
        obs["prompt"] = self.prompt
        return self._inputs(obs)

    def apply_action(self, raw_action: np.ndarray) -> None:
        action = self._outputs(raw_action)
        obs, reward, done, info = self.env.step(action)
        self._done = done
        self._last_info = info

    def is_episode_complete(self) -> bool:
        return self._done

    @property
    def last_info(self) -> dict:
        return self._last_info


def make_openpi_config():
    """Return a config dict for openpi training/serving.

    This would be used when registering the Piper task with openpi's config system.
    For now, returns a reference dict showing the expected config structure.
    """
    return {
        "task_name": "piper_color_sorting",
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "image_keys": ["base_0_rgb"],
        "image_size": IMG_SIZE,
        "action_chunk_size": 10,
        "model": "pi0_fast",  # or "pi0" for flow matching
        "description": (
            "Sort colored blocks (red, blue) into matching zones on a tabletop. "
            "The Piper 6-DOF arm with parallel-jaw gripper must pick up each block "
            "and place it in the correct color zone."
        ),
    }
