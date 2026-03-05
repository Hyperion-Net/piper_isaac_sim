"""Visualize the Piper arm using MuJoCo (works on macOS, no NVIDIA GPU needed).

Usage:
    python view_piper_mujoco.py

Controls:
    - Left-click + drag: rotate camera
    - Right-click + drag: pan camera
    - Scroll: zoom
    - Double-click: select body
    - Esc: quit
"""

import os
import math
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper.xml")

# Home pose: joint2=90° shoulder up, joint3=-90° elbow bent
HOME = [0, 1.57, -1.57, 0, 0, 0, 0.015, -0.015]


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    # Initialize to home keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        for i, q in enumerate(HOME):
            data.qpos[i] = q
            data.ctrl[i] = q
    mujoco.mj_forward(model, data)

    print(f"[INFO] Loaded Piper arm: {model.nbody} bodies, {model.njnt} joints, {model.nmesh} meshes")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  Joint {i}: {name}  range=[{model.jnt_range[i][0]:.3f}, {model.jnt_range[i][1]:.3f}]")

    step = [0]

    def controller(model, data):
        """Hold home pose with gentle motion on joint1 and joint5."""
        t = step[0] * model.opt.timestep
        for i, q in enumerate(HOME):
            data.ctrl[i] = q
        # Gentle sway
        data.ctrl[0] += 0.3 * math.sin(2 * math.pi * 0.15 * t)
        data.ctrl[4] += 0.3 * math.sin(2 * math.pi * 0.2 * t)
        step[0] += 1

    mujoco.set_mjcb_control(controller)

    print("[INFO] Launching viewer... (Esc to quit)")
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
