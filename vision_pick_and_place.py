#!/usr/bin/env mjpython
"""Vision-guided pick-and-place for the Piper arm.

Run with:  mjpython vision_pick_and_place.py

Instead of reading cube position from sim state, this script:
  1. Renders an RGB image from an overhead camera
  2. Detects the red cube via color segmentation (HSV)
  3. Deprojects the pixel centroid to a 3D world position
  4. Uses IK to pick up the cube and place it on the green zone

Camera snapshots are saved to vision_output/ at each stage.

Press R in the MuJoCo viewer to randomize cube position and run again.
Close the viewer window to quit.
"""

import os
import math
import numpy as np
import cv2
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper_tabletop.xml")

# IK parameters
IK_STEPS = 200
DAMPING = 1e-4
MAX_DPOS = 0.05

# Gripper
GRIPPER_OPEN = 0.035
GRIPPER_CLOSED = 0.0

# Heights
APPROACH_HEIGHT = 0.08
GRASP_HEIGHT = 0.0
TABLE_Z = 0.12  # cube center height on table

# Place zone (known — could also be detected via green color)
PLACE_POS = np.array([0.35, -0.1, 0.101])

# Camera image size
CAM_W, CAM_H = 640, 480

# Output directory for camera snapshots
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "vision_output")
_snap_counter = [0]


def save_snapshot(annotated, label=""):
    """Save annotated camera image to vision_output/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _snap_counter[0] += 1
    fname = f"{_snap_counter[0]:02d}_{label}.png" if label else f"{_snap_counter[0]:02d}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    cv2.imwrite(path, annotated)
    print(f"  [CAM] Saved {fname}")


# ── Vision ──────────────────────────────────────────────────────────

class OverheadCamera:
    """Renders from the overhead camera and deprojects pixels to world coords."""

    def __init__(self, model):
        self.model = model
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        self.renderer = mujoco.Renderer(model, height=CAM_H, width=CAM_W)

        # Intrinsics
        fovy_rad = math.radians(model.cam_fovy[self.cam_id])
        self.fy = CAM_H / (2 * math.tan(fovy_rad / 2))
        self.fx = self.fy
        self.cx = CAM_W / 2
        self.cy = CAM_H / 2

        # Extrinsics — build rotation from xyaxes stored in model
        # Camera x/y axes in world frame
        x_ax = np.array([-0.7682, 0.6402, 0.0])
        y_ax = np.array([-0.4835, -0.5802, 0.6554])
        z_ax = np.cross(x_ax, y_ax)
        x_ax /= np.linalg.norm(x_ax)
        y_ax /= np.linalg.norm(y_ax)
        z_ax /= np.linalg.norm(z_ax)
        self.R = np.column_stack([x_ax, y_ax, z_ax])
        self.cam_pos = model.cam_pos[self.cam_id].copy()

    def render_rgb(self, data):
        self.renderer.update_scene(data, camera=self.cam_id)
        return self.renderer.render().copy()

    def pixel_to_world(self, u, v, z_plane):
        """Ray-plane intersection: pixel (u,v) -> world point on z=z_plane."""
        ray_cam = np.array([
            (u - self.cx) / self.fx,
            -(v - self.cy) / self.fy,
            -1.0,
        ])
        ray_world = self.R @ ray_cam
        t = (z_plane - self.cam_pos[2]) / ray_world[2]
        return self.cam_pos + t * ray_world

    def detect_red_cube(self, rgb):
        """Find red cube centroid in RGB image. Returns (u, v) or None."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2

        # Clean up noise
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n_pixels = np.count_nonzero(red_mask)
        if n_pixels < 20:
            return None, red_mask

        ys, xs = np.where(red_mask > 0)
        return (float(np.mean(xs)), float(np.mean(ys))), red_mask

    def detect_cube_world(self, data):
        """Full pipeline: render -> detect -> deproject. Returns (xyz, rgb, annotated) or None."""
        rgb = self.render_rgb(data)
        centroid, mask = self.detect_red_cube(rgb)

        # Create annotated image
        annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if centroid is None:
            cv2.putText(annotated, "No cube detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, rgb, annotated

        u, v = centroid
        world_pos = self.pixel_to_world(u, v, TABLE_Z)

        # Annotate
        cu, cv_ = int(u), int(v)
        cv2.circle(annotated, (cu, cv_), 12, (0, 255, 0), 2)
        cv2.putText(annotated, f"({world_pos[0]:.3f}, {world_pos[1]:.3f})",
                    (cu + 15, cv_ - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(annotated, "VISION PICK & PLACE", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return world_pos, rgb, annotated


# ── IK & Gripper (shared with pick_and_place.py) ───────────────────

def solve_ik(model, data, target_pos, arm_joint_ids, ee_site_id):
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
        dq = J.T @ np.linalg.solve(J @ J.T + DAMPING * np.eye(3), error)
        for i, jid in enumerate(arm_joint_ids):
            ik_data.qpos[jid] += dq[i]
            ik_data.qpos[jid] = np.clip(ik_data.qpos[jid], *model.jnt_range[jid])
    mujoco.mj_forward(model, ik_data)
    if np.linalg.norm(target_pos - ik_data.site_xpos[ee_site_id]) < 0.02:
        return np.array([ik_data.qpos[jid] for jid in arm_joint_ids])
    return None


def set_gripper(model, data, value):
    data.ctrl[6] = value
    data.ctrl[7] = -value
    grasp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    if grasp_id >= 0:
        if value == GRIPPER_CLOSED:
            gb_id = model.eq_obj1id[grasp_id]
            cube_id = model.eq_obj2id[grasp_id]
            gb_pos = data.xpos[gb_id]
            gb_mat = data.xmat[gb_id].reshape(3, 3)
            rel_pos = gb_mat.T @ (data.xpos[cube_id] - gb_pos)
            model.eq_data[grasp_id, 3:6] = rel_pos
            model.eq_data[grasp_id, 6:10] = [1, 0, 0, 0]
            data.eq_active[grasp_id] = 1
        else:
            data.eq_active[grasp_id] = 0


def wait_steps(model, data, viewer, n_steps):
    for _ in range(n_steps):
        if not viewer.is_running():
            return False
        mujoco.mj_step(model, data)
    viewer.sync()
    return True


def move_to(model, data, viewer, target_pos, arm_joint_ids, ee_site_id, settle_steps=800):
    joint_targets = solve_ik(model, data, target_pos, arm_joint_ids, ee_site_id)
    if joint_targets is None:
        print(f"    WARNING: IK failed for target {target_pos}")
        return True
    for i, jid in enumerate(arm_joint_ids):
        data.qpos[jid] = joint_targets[i]
        data.qvel[jid] = 0.0
        data.ctrl[jid] = joint_targets[i]
    for step in range(settle_steps):
        if not viewer.is_running():
            return False
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            viewer.sync()
    viewer.sync()
    return True


# ── Main ────────────────────────────────────────────────────────────

def run_vision_pick_and_place(model, data, viewer, camera, arm_joint_ids, ee_site_id):
    """Detect cube via vision, then pick and place."""

    # Step 1: Capture image and detect cube
    print("  [VISION] Capturing overhead image...")
    mujoco.mj_forward(model, data)
    result, rgb, annotated = camera.detect_cube_world(data)
    save_snapshot(annotated, "detect")

    if result is None:
        print("  [VISION] No red cube detected!")
        return True

    cube_xy = result[:2]
    cube_pos = np.array([cube_xy[0], cube_xy[1], TABLE_Z])
    print(f"  [VISION] Cube detected at ({cube_pos[0]:.3f}, {cube_pos[1]:.3f})")

    # Compare with ground truth
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    actual = data.xpos[cube_body_id]
    err = np.linalg.norm(cube_pos[:2] - actual[:2])
    print(f"  [VISION] Ground truth: ({actual[0]:.3f}, {actual[1]:.3f}), error: {err:.4f}m")

    # Step 2: Pick and place using vision-detected position
    pick_above = cube_pos.copy()
    pick_above[2] += APPROACH_HEIGHT
    pick_at = cube_pos.copy()
    pick_at[2] += GRASP_HEIGHT

    place_above = PLACE_POS.copy()
    place_above[2] += APPROACH_HEIGHT
    place_at = PLACE_POS.copy()
    place_at[2] += GRASP_HEIGHT

    print("  [1/7] Opening gripper...")
    set_gripper(model, data, GRIPPER_OPEN)
    if not wait_steps(model, data, viewer, 200):
        return False

    print("  [2/7] Approaching cube...")
    if not move_to(model, data, viewer, pick_above, arm_joint_ids, ee_site_id):
        return False
    _, _, ann = camera.detect_cube_world(data)
    save_snapshot(ann, "approach")

    print("  [3/7] Descending to grasp...")
    if not move_to(model, data, viewer, pick_at, arm_joint_ids, ee_site_id):
        return False

    print("  [4/7] Closing gripper...")
    set_gripper(model, data, GRIPPER_CLOSED)
    if not wait_steps(model, data, viewer, 1000):
        return False

    print("  [5/7] Lifting...")
    if not move_to(model, data, viewer, pick_above, arm_joint_ids, ee_site_id, settle_steps=1500):
        return False
    _, _, ann = camera.detect_cube_world(data)
    save_snapshot(ann, "lift")

    print("  [6/7] Moving to place zone...")
    if not move_to(model, data, viewer, place_above, arm_joint_ids, ee_site_id):
        return False

    print("  [7/7] Placing...")
    if not move_to(model, data, viewer, place_at, arm_joint_ids, ee_site_id):
        return False

    set_gripper(model, data, GRIPPER_OPEN)
    if not wait_steps(model, data, viewer, 300):
        return False

    if not move_to(model, data, viewer, place_above, arm_joint_ids, ee_site_id):
        return False

    # Final camera snapshot
    _, _, ann = camera.detect_cube_world(data)
    save_snapshot(ann, "done")

    # Check success
    mujoco.mj_forward(model, data)
    final_pos = data.xpos[cube_body_id].copy()
    dist = np.linalg.norm(final_pos[:2] - PLACE_POS[:2])
    if dist < 0.05:
        print(f"\n  SUCCESS! Cube placed {dist:.3f}m from target.")
    else:
        print(f"\n  Cube landed {dist:.3f}m from target.")

    return True


def randomize_cube(data, rng):
    """Place cube at a random position on the table."""
    x = rng.uniform(0.2, 0.5)
    y = rng.uniform(-0.15, 0.2)
    data.qpos[8:11] = [x, y, TABLE_Z]
    data.qpos[11:15] = [1, 0, 0, 0]
    # Zero cube velocity
    data.qvel[8:14] = 0


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    camera = OverheadCamera(model)
    rng = np.random.default_rng(42)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_joint_ids = list(range(6))

    reset_requested = [False]

    def key_callback(keycode):
        if keycode == 82:  # R
            reset_requested[0] = True

    def reset_sim(randomize=False):
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        if randomize:
            randomize_cube(data, rng)
        mujoco.mj_forward(model, data)
        set_gripper(model, data, GRIPPER_OPEN)

    reset_sim()

    print("=" * 50)
    print("  PIPER ARM — VISION PICK AND PLACE")
    print("=" * 50)
    print()
    print("  The arm uses an overhead camera to detect")
    print("  the red cube, then picks and places it.")
    print(f"  Snapshots saved to vision_output/")
    print()
    print("  Press R to randomize cube and run again.")
    print("  Close window to quit.")
    print()

    viewer = mujoco.viewer.launch_passive(
        model, data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )

    # Clear previous snapshots
    if os.path.isdir(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".png"):
                os.remove(os.path.join(OUTPUT_DIR, f))
    _snap_counter[0] = 0

    wait_steps(model, data, viewer, 200)

    while viewer.is_running():
        print("\n--- Starting vision pick-and-place ---\n")
        run_vision_pick_and_place(model, data, viewer, camera, arm_joint_ids, ee_site_id)

        print("\n  Press R to randomize cube and run again.")
        while viewer.is_running() and not reset_requested[0]:
            mujoco.mj_step(model, data)
            viewer.sync()

        if reset_requested[0]:
            reset_requested[0] = False
            reset_sim(randomize=True)
            wait_steps(model, data, viewer, 200)
            _, _, ann = camera.detect_cube_world(data)
            save_snapshot(ann, "reset")
            print("\n  Reset with new cube position!")

    viewer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
