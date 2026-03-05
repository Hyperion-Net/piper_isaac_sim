#!/usr/bin/env mjpython
"""Color sorting demo for the Piper arm.

Run with:  mjpython sort_blocks.py [--mode scripted|policy] [--host HOST] [--port PORT]

Modes:
  scripted (default) — Automated IK-based sorting (no GPU needed)
  policy             — Uses openpi policy server for actions

Press R to reset and run again. Close window to quit.
"""

import os
import sys
import argparse
import numpy as np
import mujoco
import mujoco.viewer

from sorting_env import (
    PiperSortingEnv,
    BLOCK_NAMES,
    BLOCK_COLORS,
    ZONE_POSITIONS,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
    APPROACH_HEIGHT,
    GRASP_HEIGHT,
    SORT_TOLERANCE,
    solve_ik,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_gripper(model, data, env, value, block_idx=None):
    """Set gripper and manage grasp weld."""
    data.ctrl[6] = value
    data.ctrl[7] = -value
    if value == GRIPPER_CLOSED and block_idx is not None:
        env._try_grasp()
    elif value == GRIPPER_OPEN:
        env._release()


def wait_steps(model, data, viewer, n_steps):
    for _ in range(n_steps):
        if not viewer.is_running():
            return False
        mujoco.mj_step(model, data)
    viewer.sync()
    return True


def move_to(model, data, viewer, env, target_pos, settle_steps=800):
    """IK solve and move arm to target."""
    joint_targets = solve_ik(model, data, target_pos, env.arm_joint_ids, env.ee_site_id)
    if joint_targets is None:
        print(f"    WARNING: IK failed for target {target_pos}")
        return True

    for i, jid in enumerate(env.arm_joint_ids):
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


def find_nearest_unsorted_block(env):
    """Find the index of the nearest unsorted block to sort next."""
    mujoco.mj_forward(env.model, env.data)
    unsorted = []
    for i, body_id in enumerate(env.block_body_ids):
        pos = env.data.xpos[body_id].copy()
        color = BLOCK_COLORS[i]
        zone_pos = ZONE_POSITIONS[color]
        if np.linalg.norm(pos[:2] - zone_pos[:2]) >= SORT_TOLERANCE:
            unsorted.append((i, pos))
    if not unsorted:
        return None, None

    # Pick the one closest to the end-effector
    ee_pos = env.data.site_xpos[env.ee_site_id]
    unsorted.sort(key=lambda x: np.linalg.norm(x[1] - ee_pos))
    return unsorted[0]


def run_scripted_sorting(env, viewer):
    """Sort all blocks using scripted IK motions."""
    model = env.model
    data = env.data

    sorted_count = 0
    total = len(BLOCK_NAMES)

    while sorted_count < total:
        block_idx, block_pos = find_nearest_unsorted_block(env)
        if block_idx is None:
            break

        color = BLOCK_COLORS[block_idx]
        zone_pos = ZONE_POSITIONS[color]
        name = BLOCK_NAMES[block_idx]

        print(f"\n  Sorting {name} ({color}) -> {color} zone")

        # Open gripper
        set_gripper(model, data, env, GRIPPER_OPEN)
        if not wait_steps(model, data, viewer, 200):
            return False

        # Approach above block
        above = block_pos.copy()
        above[2] = block_pos[2] + APPROACH_HEIGHT
        print(f"    Approaching above ({block_pos[0]:.2f}, {block_pos[1]:.2f})...")
        if not move_to(model, data, viewer, env, above):
            return False

        # Descend to grasp
        grasp_pos = block_pos.copy()
        grasp_pos[2] = block_pos[2] + GRASP_HEIGHT
        print(f"    Descending...")
        if not move_to(model, data, viewer, env, grasp_pos):
            return False

        # Close gripper
        print(f"    Grasping...")
        set_gripper(model, data, env, GRIPPER_CLOSED, block_idx=block_idx)
        if not wait_steps(model, data, viewer, 1000):
            return False

        # Lift
        print(f"    Lifting...")
        if not move_to(model, data, viewer, env, above, settle_steps=1500):
            return False

        # Move above place zone
        place_above = zone_pos.copy()
        place_above[2] = zone_pos[2] + APPROACH_HEIGHT
        print(f"    Moving to {color} zone...")
        if not move_to(model, data, viewer, env, place_above):
            return False

        # Lower and release
        place_at = zone_pos.copy()
        place_at[2] = zone_pos[2] + GRASP_HEIGHT
        print(f"    Placing...")
        if not move_to(model, data, viewer, env, place_at):
            return False

        set_gripper(model, data, env, GRIPPER_OPEN)
        if not wait_steps(model, data, viewer, 300):
            return False

        # Retreat
        if not move_to(model, data, viewer, env, place_above):
            return False

        # Check
        mujoco.mj_forward(model, data)
        final_pos = data.xpos[env.block_body_ids[block_idx]].copy()
        dist = np.linalg.norm(final_pos[:2] - zone_pos[:2])
        if dist < SORT_TOLERANCE:
            sorted_count += 1
            print(f"    Placed! ({dist:.3f}m from center) [{sorted_count}/{total}]")
        else:
            print(f"    Missed ({dist:.3f}m from center)")

    if sorted_count == total:
        print(f"\n  ALL {total} BLOCKS SORTED!")
    else:
        print(f"\n  Sorted {sorted_count}/{total} blocks.")

    return True


def run_policy_sorting(env, viewer, host, port):
    """Sort blocks using an openpi policy server."""
    try:
        from openpi_client import websocket_client_policy as wcp
    except ImportError:
        print("\n  ERROR: openpi_client not installed.")
        print("  Install with: pip install openpi-client")
        print("  Or run with --mode scripted for the IK demo.")
        return False

    from piper_openpi import PiperSortingOpenPIEnv

    openpi_env = PiperSortingOpenPIEnv()
    openpi_env.env = env  # share the same underlying env

    print(f"  Connecting to policy server at {host}:{port}...")
    policy = wcp.WebsocketClientPolicy(host=host, port=port)

    obs = openpi_env.reset()
    step = 0
    max_steps = 4000

    while not openpi_env.is_episode_complete() and step < max_steps:
        if not viewer.is_running():
            return False

        # Get action from policy
        raw_action = policy.infer(obs)["actions"]
        openpi_env.apply_action(raw_action)

        # Simulate and sync viewer
        for _ in range(5):
            mujoco.mj_step(env.model, env.data)
        viewer.sync()

        obs = openpi_env.get_observation()
        step += 1

        if step % 100 == 0:
            info = openpi_env.last_info
            print(f"    Step {step}: {info.get('sorted_count', 0)}/{info.get('total_blocks', 4)} sorted")

    info = openpi_env.last_info
    print(f"\n  Done! {info.get('sorted_count', 0)}/{info.get('total_blocks', 4)} blocks sorted in {step} steps.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Piper color sorting demo")
    parser.add_argument("--mode", choices=["scripted", "policy"], default="scripted",
                        help="scripted=IK demo, policy=openpi server")
    parser.add_argument("--host", default="localhost", help="openpi server host")
    parser.add_argument("--port", type=int, default=8000, help="openpi server port")
    parser.add_argument("--no-randomize", action="store_true", help="Use fixed block positions")
    args = parser.parse_args()

    env = PiperSortingEnv()
    env.reset(randomize=not args.no_randomize)

    reset_requested = [False]

    def key_callback(keycode):
        if keycode == 82:  # R
            reset_requested[0] = True

    print("=" * 50)
    print("  PIPER ARM — COLOR SORTING")
    print("=" * 50)
    print()
    print("  Sort red blocks to the red zone,")
    print("  blue blocks to the blue zone.")
    print()
    print(f"  Mode: {args.mode}")
    print("  Press R to reset. Close window to quit.")
    print()

    viewer = mujoco.viewer.launch_passive(
        env.model, env.data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )

    wait_steps(env.model, env.data, viewer, 200)

    while viewer.is_running():
        print("\n--- Starting sorting sequence ---\n")

        if args.mode == "scripted":
            run_scripted_sorting(env, viewer)
        else:
            run_policy_sorting(env, viewer, args.host, args.port)

        print("\n  Press R to reset and run again, or close window to quit.")
        while viewer.is_running() and not reset_requested[0]:
            mujoco.mj_step(env.model, env.data)
            viewer.sync()

        if reset_requested[0]:
            reset_requested[0] = False
            env.reset(randomize=not args.no_randomize)
            wait_steps(env.model, env.data, viewer, 200)
            print("\n  Reset complete!")

    viewer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
