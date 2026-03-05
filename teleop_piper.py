#!/usr/bin/env mjpython
"""Interactive joint control for the Piper arm in MuJoCo.

Run with:  mjpython teleop_piper.py
    (or)   python teleop_piper.py   (on Linux)

Keyboard Controls:
    1-6      Select arm joint (joint1-joint6)
    7        Select gripper
    UP/DOWN  Increase/decrease selected joint angle
    +/-      Increase/decrease step size
    H        Reset to home pose
    0        Reset to zero pose
    SPACE    Print current joint angles
    Esc      Quit

Camera:
    Left-click + drag   Rotate
    Right-click + drag  Pan
    Scroll              Zoom
"""

import os
import time
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper.xml")

HOME = [0, 1.57, -1.57, 0, 0, 0, 0.015, -0.015]

# GLFW key codes
KEY_UP = 265
KEY_DOWN = 264
KEY_SPACE = 32
KEY_EQUAL = 61
KEY_MINUS = 45
KEY_H = 72
KEY_0 = 48
KEY_1 = 49

JOINT_LABELS = [
    "Joint 1  (base yaw)",
    "Joint 2  (shoulder)",
    "Joint 3  (elbow)",
    "Joint 4  (wrist yaw)",
    "Joint 5  (wrist pitch)",
    "Joint 6  (wrist roll)",
    "Gripper   (open/close)",
]


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # Start at home pose
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # State
    selected = [0]
    step_size = [0.05]
    targets = list(HOME)

    def print_status():
        sel = selected[0]
        print(f"\n--- Piper Arm Teleop ---")
        print(f"Step size: {step_size[0]:.3f} rad")
        for i, label in enumerate(JOINT_LABELS):
            marker = " >> " if i == sel else "    "
            if i < 6:
                val = targets[i]
                lo, hi = model.jnt_range[i]
                print(f"{marker}{label}:  {val:+.3f}  [{lo:.2f}, {hi:.2f}]")
            else:
                grip = targets[6]
                print(f"{marker}{label}:  {grip:.4f}  [0, 0.035]")

    def apply_targets():
        for i in range(8):
            data.ctrl[i] = targets[i]

    def key_callback(keycode):
        sel = selected[0]

        # Joint selection: 1-7
        if KEY_1 <= keycode <= KEY_1 + 6:
            selected[0] = keycode - KEY_1
            print_status()
            return

        # Adjust selected joint
        if keycode in (KEY_UP, KEY_DOWN):
            if sel < 6:
                delta = step_size[0] if keycode == KEY_UP else -step_size[0]
                lo, hi = model.jnt_range[sel]
                targets[sel] = max(lo, min(hi, targets[sel] + delta))
            else:
                grip_delta = 0.002 if keycode == KEY_UP else -0.002
                targets[6] = max(0, min(0.035, targets[6] + grip_delta))
                targets[7] = -targets[6]

            apply_targets()
            name = JOINT_LABELS[sel].split("(")[0].strip()
            val = targets[sel] if sel < 6 else targets[6]
            print(f"  {name}: {val:+.4f}")
            return

        # Step size
        if keycode == KEY_EQUAL:
            step_size[0] = min(0.5, step_size[0] * 2)
            print(f"  Step size: {step_size[0]:.3f} rad")
        elif keycode == KEY_MINUS:
            step_size[0] = max(0.005, step_size[0] / 2)
            print(f"  Step size: {step_size[0]:.3f} rad")

        # Home
        elif keycode == KEY_H:
            for i, q in enumerate(HOME):
                targets[i] = q
            apply_targets()
            print("  >> HOME")
            print_status()

        # Zero
        elif keycode == KEY_0:
            for i in range(8):
                targets[i] = 0.0
            apply_targets()
            print("  >> ZERO")
            print_status()

        # Print angles
        elif keycode == KEY_SPACE:
            print_status()
            angles = [f"{t:.4f}" for t in targets[:6]]
            print(f"\n  Config: [{', '.join(angles)}]  gripper: {targets[6]:.4f}")

    # Apply initial targets
    apply_targets()

    print("=" * 50)
    print("  PIPER ARM INTERACTIVE TELEOP")
    print("=" * 50)
    print()
    print("  Keys 1-6 : select arm joint")
    print("  Key 7    : select gripper")
    print("  UP/DOWN  : adjust selected joint")
    print("  +/-      : change step size")
    print("  H        : home pose")
    print("  0        : zero pose")
    print("  SPACE    : print joint config")
    print("  Esc      : quit")
    print()
    print_status()

    # Use launch_passive (requires mjpython on macOS)
    viewer = mujoco.viewer.launch_passive(
        model, data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=True,
    )

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
