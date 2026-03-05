#!/usr/bin/env mjpython
"""Pick-and-place demo for the Piper arm.

Run with:  mjpython pick_and_place.py

The arm automatically:
  1. Reaches above the red cube
  2. Descends to grasp
  3. Closes gripper
  4. Lifts cube
  5. Moves over the green place zone
  6. Lowers and releases
  7. Retreats upward

Press R to reset and run again.
Close the window to quit.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper_tabletop.xml")

HOME = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0, 0.035, -0.035])

# IK parameters
IK_STEPS = 200
DAMPING = 1e-4
MAX_DPOS = 0.05

# Gripper — cube is 0.04m wide, need >0.04m total opening
GRIPPER_OPEN = 0.035    # max opening per finger (0.07m total gap)
GRIPPER_CLOSED = 0.0

# Task positions (relative to world)
CUBE_POS = np.array([0.35, 0.08, 0.12])
PLACE_POS = np.array([0.35, -0.1, 0.101])
APPROACH_HEIGHT = 0.08   # height above target for approach/retreat
GRASP_HEIGHT = 0.0       # ee_site at cube center so fingers wrap around it


def solve_ik(model, data, target_pos, arm_joint_ids, ee_site_id):
    """Damped least-squares IK on a scratch copy. Returns joint angles (or None)."""
    ik_data = mujoco.MjData(model)
    # Copy current arm joint positions
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


def set_gripper(model, data, value):
    """Set gripper opening and manage grasp weld constraint."""
    data.ctrl[6] = value
    data.ctrl[7] = -value
    grasp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp")
    if grasp_id >= 0:
        if value == GRIPPER_CLOSED:
            # Compute current relative pose of cube w.r.t. gripper_base
            gb_id = model.eq_obj1id[grasp_id]
            cube_id = model.eq_obj2id[grasp_id]
            gb_pos = data.xpos[gb_id]
            gb_mat = data.xmat[gb_id].reshape(3, 3)
            cube_pos = data.xpos[cube_id]
            rel_pos = gb_mat.T @ (cube_pos - gb_pos)
            # eq_data layout for weld: [anchor(3), relpose_pos(3), relpose_quat(4), torquescale(1)]
            model.eq_data[grasp_id, 3:6] = rel_pos
            model.eq_data[grasp_id, 6:10] = [1, 0, 0, 0]
            data.eq_active[grasp_id] = 1
        else:
            data.eq_active[grasp_id] = 0


def wait_steps(model, data, viewer, n_steps):
    """Simulate n_steps and sync viewer. Returns False if viewer closed."""
    for _ in range(n_steps):
        if not viewer.is_running():
            return False
        mujoco.mj_step(model, data)
    viewer.sync()
    return True


def move_to(model, data, viewer, target_pos, arm_joint_ids, ee_site_id, settle_steps=800):
    """IK solve, set arm joints + actuators, then simulate to settle."""
    joint_targets = solve_ik(model, data, target_pos, arm_joint_ids, ee_site_id)
    if joint_targets is None:
        print(f"    WARNING: IK failed for target {target_pos}")
        return True  # continue anyway

    # Set both qpos and ctrl so the arm starts at the IK solution
    # (only arm joints — cube freejoint is untouched)
    for i, jid in enumerate(arm_joint_ids):
        data.qpos[jid] = joint_targets[i]
        data.qvel[jid] = 0.0
        data.ctrl[jid] = joint_targets[i]

    # Simulate to let physics settle (gripper contact, cube dynamics)
    for step in range(settle_steps):
        if not viewer.is_running():
            return False
        mujoco.mj_step(model, data)
        if step % 5 == 0:
            viewer.sync()

    viewer.sync()
    return True


def run_pick_and_place(model, data, viewer, arm_joint_ids, ee_site_id):
    """Execute one pick-and-place cycle."""
    # Read actual cube position from simulation
    cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_body_id].copy()

    pick_above = cube_pos.copy()
    pick_above[2] = cube_pos[2] + APPROACH_HEIGHT

    pick_at = cube_pos.copy()
    pick_at[2] = cube_pos[2] + GRASP_HEIGHT

    place_above = PLACE_POS.copy()
    place_above[2] = PLACE_POS[2] + APPROACH_HEIGHT

    place_at = PLACE_POS.copy()
    place_at[2] = PLACE_POS[2] + GRASP_HEIGHT

    # 1. Open gripper
    print("  [1/7] Opening gripper...")
    set_gripper(model, data, GRIPPER_OPEN)
    if not wait_steps(model, data, viewer, 200):
        return False

    # 2. Move above cube
    print(f"  [2/7] Approaching cube at ({cube_pos[0]:.2f}, {cube_pos[1]:.2f})...")
    if not move_to(model, data, viewer, pick_above, arm_joint_ids, ee_site_id):
        return False

    # 3. Descend to grasp
    print("  [3/7] Descending to grasp...")
    if not move_to(model, data, viewer, pick_at, arm_joint_ids, ee_site_id):
        return False

    # 4. Close gripper
    print("  [4/7] Closing gripper...")
    set_gripper(model, data, GRIPPER_CLOSED)
    if not wait_steps(model, data, viewer, 1000):
        return False

    # 5. Lift slowly
    print("  [5/7] Lifting...")
    if not move_to(model, data, viewer, pick_above, arm_joint_ids, ee_site_id, settle_steps=1500):
        return False

    # 6. Move above place zone
    print(f"  [6/7] Moving to place zone ({PLACE_POS[0]:.2f}, {PLACE_POS[1]:.2f})...")
    if not move_to(model, data, viewer, place_above, arm_joint_ids, ee_site_id):
        return False

    # 7. Lower and release
    print("  [7/7] Placing...")
    if not move_to(model, data, viewer, place_at, arm_joint_ids, ee_site_id):
        return False

    set_gripper(model, data, GRIPPER_OPEN)
    if not wait_steps(model, data, viewer, 300):
        return False

    # Retreat upward
    if not move_to(model, data, viewer, place_above, arm_joint_ids, ee_site_id):
        return False

    # Check success: is cube near the place zone?
    mujoco.mj_forward(model, data)
    final_cube_pos = data.xpos[cube_body_id].copy()
    dist = np.linalg.norm(final_cube_pos[:2] - PLACE_POS[:2])
    if dist < 0.05:
        print(f"\n  SUCCESS! Cube placed {dist:.3f}m from target center.")
    else:
        print(f"\n  Cube landed {dist:.3f}m from target (>{0.05:.2f}m).")

    return True


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    arm_joint_ids = list(range(6))

    reset_requested = [False]

    def key_callback(keycode):
        if keycode == 82:  # R
            reset_requested[0] = True

    def reset_sim():
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        set_gripper(model, data, GRIPPER_OPEN)

    # Initial reset
    reset_sim()

    print("=" * 50)
    print("  PIPER ARM — PICK AND PLACE")
    print("=" * 50)
    print()
    print("  The arm will pick up the red cube")
    print("  and place it on the green zone.")
    print()
    print("  Press R to reset and run again.")
    print("  Close window to quit.")
    print()

    viewer = mujoco.viewer.launch_passive(
        model, data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )

    # Let the scene settle before starting
    wait_steps(model, data, viewer, 200)

    while viewer.is_running():
        print("\n--- Starting pick-and-place sequence ---\n")
        run_pick_and_place(model, data, viewer, arm_joint_ids, ee_site_id)

        # Wait for reset or close
        print("\n  Press R to reset and run again, or close window to quit.")
        while viewer.is_running() and not reset_requested[0]:
            mujoco.mj_step(model, data)
            viewer.sync()

        if reset_requested[0]:
            reset_requested[0] = False
            reset_sim()
            wait_steps(model, data, viewer, 200)
            print("\n  Reset complete!")

    viewer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
