"""Color sorting environment for the Piper arm.

Provides a Gymnasium-style environment and an openpi-compatible Environment class.
The task: sort red blocks to the red zone and blue blocks to the blue zone.

Can be used standalone (scripted demo) or with an openpi policy server.
"""

import os
import numpy as np
import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper_sorting.xml")

# Block and zone definitions
BLOCK_NAMES = ["red_block_0", "red_block_1", "blue_block_0", "blue_block_1"]
BLOCK_COLORS = ["red", "red", "blue", "blue"]
ZONE_POSITIONS = {
    "red": np.array([0.45, -0.2, 0.101]),
    "blue": np.array([0.25, -0.2, 0.101]),
}
GRASP_NAMES = ["grasp_red_0", "grasp_red_1", "grasp_blue_0", "grasp_blue_1"]

# IK parameters
IK_STEPS = 200
DAMPING = 1e-4
MAX_DPOS = 0.05

# Gripper
GRIPPER_OPEN = 0.035
GRIPPER_CLOSED = 0.0

# Task geometry
APPROACH_HEIGHT = 0.08
GRASP_HEIGHT = 0.0
SORT_TOLERANCE = 0.05  # block within 5cm of zone center = sorted


class PiperSortingEnv:
    """MuJoCo environment for the color sorting task."""

    def __init__(self, render_size=(224, 224)):
        self.model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        self.data = mujoco.MjData(self.model)
        self.render_size = render_size

        # Cache IDs
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.arm_joint_ids = list(range(6))
        self.overhead_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")

        self.block_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in BLOCK_NAMES
        ]
        self.grasp_eq_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            for name in GRASP_NAMES
        ]

        self._active_grasp = None  # index of currently grasped block
        self._renderer = None
        self._step_count = 0
        self._max_steps = 4000

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, *self.render_size)
        return self._renderer

    def reset(self, randomize=True):
        """Reset to home keyframe, optionally randomize block positions."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Deactivate all grasp constraints
        for eq_id in self.grasp_eq_ids:
            self.data.eq_active[eq_id] = 0
        self._active_grasp = None

        if randomize:
            self._randomize_blocks()

        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self.get_observation()

    def _randomize_blocks(self):
        """Place blocks at random positions on the table top half (y > 0)."""
        table_x_range = (0.15, 0.55)
        table_y_range = (0.0, 0.25)
        z = 0.125  # just above table surface

        positions = []
        for i, body_id in enumerate(self.block_body_ids):
            # Find non-overlapping position
            for _ in range(50):
                x = np.random.uniform(*table_x_range)
                y = np.random.uniform(*table_y_range)
                pos = np.array([x, y, z])
                if all(np.linalg.norm(pos[:2] - p[:2]) > 0.06 for p in positions):
                    positions.append(pos)
                    break
            else:
                positions.append(pos)

            # freejoint qpos: first 8 are arm joints, then 7 per block (xyz + quat)
            qpos_start = 8 + i * 7
            self.data.qpos[qpos_start:qpos_start + 3] = positions[-1]
            self.data.qpos[qpos_start + 3:qpos_start + 7] = [1, 0, 0, 0]

    def get_observation(self):
        """Return observation dict compatible with openpi format.

        Returns:
            dict with keys:
                "state": np.array of shape (8,) — 6 arm joints + 2 gripper joints
                "images": {"overhead_0_rgb": uint8 array (H, W, 3)}
                "block_positions": np.array (4, 3) — world positions of all blocks
        """
        mujoco.mj_forward(self.model, self.data)

        # Joint state (arm + gripper)
        state = np.array([self.data.qpos[i] for i in range(8)], dtype=np.float32)

        # Overhead camera image
        self.renderer.update_scene(self.data, camera=self.overhead_cam_id)
        rgb = self.renderer.render().copy()

        # Block positions (ground truth, for reward/evaluation)
        block_positions = np.array([
            self.data.xpos[bid].copy() for bid in self.block_body_ids
        ])

        return {
            "state": state,
            "images": {"overhead_0_rgb": rgb},
            "block_positions": block_positions,
        }

    def step(self, action):
        """Execute one action step.

        Args:
            action: np.array of shape (8,) — target positions for 6 arm joints + 2 gripper joints
                    OR np.array of shape (9,) — same + grasp_trigger (-1=release, 0=no change, 1=grasp)

        Returns:
            obs, reward, done, info
        """
        self._step_count += 1

        # Parse action
        joint_targets = action[:8]
        grasp_trigger = action[8] if len(action) > 8 else 0.0

        # Set actuator targets
        for i in range(8):
            self.data.ctrl[i] = joint_targets[i]

        # Handle grasp trigger
        if grasp_trigger > 0.5:
            self._try_grasp()
        elif grasp_trigger < -0.5:
            self._release()

        # Simulate
        for _ in range(10):  # 10 steps at 0.002s = 20ms per action
            mujoco.mj_step(self.model, self.data)

        obs = self.get_observation()
        reward = self._compute_reward(obs["block_positions"])
        done = self._check_done(obs["block_positions"])

        if self._step_count >= self._max_steps:
            done = True

        info = {
            "sorted_count": self._count_sorted(obs["block_positions"]),
            "total_blocks": len(BLOCK_NAMES),
            "step": self._step_count,
        }

        return obs, reward, done, info

    def _try_grasp(self):
        """Activate weld on the nearest block if gripper is near it."""
        if self._active_grasp is not None:
            return  # already holding something

        ee_pos = self.data.site_xpos[self.ee_site_id]
        for i, body_id in enumerate(self.block_body_ids):
            block_pos = self.data.xpos[body_id]
            dist = np.linalg.norm(ee_pos - block_pos)
            if dist < 0.04:
                eq_id = self.grasp_eq_ids[i]
                # Compute relative pose
                gb_id = self.model.eq_obj1id[eq_id]
                gb_pos = self.data.xpos[gb_id]
                gb_mat = self.data.xmat[gb_id].reshape(3, 3)
                rel_pos = gb_mat.T @ (block_pos - gb_pos)
                self.model.eq_data[eq_id, 3:6] = rel_pos
                self.model.eq_data[eq_id, 6:10] = [1, 0, 0, 0]
                self.data.eq_active[eq_id] = 1
                self._active_grasp = i
                break

    def _release(self):
        """Deactivate current grasp weld."""
        if self._active_grasp is not None:
            eq_id = self.grasp_eq_ids[self._active_grasp]
            self.data.eq_active[eq_id] = 0
            self._active_grasp = None

    def _compute_reward(self, block_positions):
        """Reward based on how many blocks are in their correct zone."""
        reward = 0.0
        for i, pos in enumerate(block_positions):
            color = BLOCK_COLORS[i]
            zone_pos = ZONE_POSITIONS[color]
            dist = np.linalg.norm(pos[:2] - zone_pos[:2])
            if dist < SORT_TOLERANCE:
                reward += 1.0
            else:
                # Shaping: closer = better
                reward -= 0.1 * dist
        return reward

    def _count_sorted(self, block_positions):
        count = 0
        for i, pos in enumerate(block_positions):
            color = BLOCK_COLORS[i]
            zone_pos = ZONE_POSITIONS[color]
            if np.linalg.norm(pos[:2] - zone_pos[:2]) < SORT_TOLERANCE:
                count += 1
        return count

    def _check_done(self, block_positions):
        return self._count_sorted(block_positions) == len(BLOCK_NAMES)


# --- IK utilities (reused from pick_and_place.py) ---

def solve_ik(model, data, target_pos, arm_joint_ids, ee_site_id):
    """Damped least-squares IK on a scratch copy."""
    ik_data = mujoco.MjData(model)
    for jid in arm_joint_ids:
        ik_data.qpos[jid] = data.qpos[jid]

    jacp = np.zeros((3, model.nv))

    for _ in range(IK_STEPS):
        mujoco.mj_forward(model, ik_data)
        error = target_pos - ik_data.site_xpos[ee_site_id]

        if np.linalg.norm(error) < 5e-4:
            return np.array([ik_data.qpos[jid] for jid in arm_joint_ids])

        err_norm = np.linalg.norm(error)
        if err_norm > MAX_DPOS:
            error = error * MAX_DPOS / err_norm

        mujoco.mj_jacSite(model, ik_data, jacp, None, ee_site_id)
        J = jacp[:, arm_joint_ids]
        JJT = J @ J.T + DAMPING * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, error)

        for i, jid in enumerate(arm_joint_ids):
            ik_data.qpos[jid] += dq[i]
            lo, hi = model.jnt_range[jid]
            ik_data.qpos[jid] = np.clip(ik_data.qpos[jid], lo, hi)

    mujoco.mj_forward(model, ik_data)
    err = np.linalg.norm(target_pos - ik_data.site_xpos[ee_site_id])
    if err < 0.02:
        return np.array([ik_data.qpos[jid] for jid in arm_joint_ids])
    return None
