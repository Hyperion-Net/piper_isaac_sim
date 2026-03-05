#!/usr/bin/env mjpython
"""Click-to-reach IK for the Piper arm.

Run with:  mjpython ik_teleop_piper.py

How to use:
    1. Double-click the GREEN BALL to select it (turns yellow)
    2. Double-click anywhere on the floor to send it there
    3. Arm goes directly to the new position via IK
    4. Repeat anytime!

    Close the window to quit.
"""

import os
import numpy as np
import mujoco
import mujoco.viewer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "mjcf", "piper.xml")

HOME = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0, 0.015, -0.015])

IK_STEPS = 100
DAMPING = 1e-4
MAX_DPOS = 0.05
DEFAULT_HEIGHT = 0.25


def main():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ik_target")
    target_geom_id = model.body_geomadr[target_body_id]
    arm_joint_ids = list(range(6))

    arm_body_ids = set()
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name not in ("world", "ik_target"):
            arm_body_ids.add(i)

    # Start at home
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    data.mocap_pos[0] = data.site_xpos[ee_site_id].copy()

    COLOR_IDLE = np.array([0.1, 0.9, 0.2, 0.7])
    COLOR_SELECTED = np.array([1.0, 0.9, 0.0, 0.9])
    COLOR_MOVING = np.array([1.0, 0.5, 0.0, 0.7])

    ball_selected = False
    last_select = -1
    reaching = False
    jacp = np.zeros((3, model.nv))

    print("=" * 50)
    print("  PIPER ARM — CLICK-TO-REACH IK")
    print("=" * 50)
    print()
    print("  1. Double-click the GREEN BALL")
    print("  2. Double-click where you want it to go")
    print("  3. Arm goes directly there. Repeat!")
    print()

    viewer = mujoco.viewer.launch_passive(
        model, data,
        show_left_ui=False,
        show_right_ui=False,
    )

    while viewer.is_running():
        sel = viewer.perturb.select

        if sel != last_select:
            last_select = sel

            if sel == target_body_id:
                ball_selected = True
                model.geom_rgba[target_geom_id] = COLOR_SELECTED
                print("  Ball selected — double-click a destination")

            elif ball_selected and sel not in arm_body_ids:
                local = viewer.perturb.localpos.copy()
                if sel == 0:
                    world_pos = local.copy()
                    world_pos[2] = DEFAULT_HEIGHT
                else:
                    body_pos = data.xpos[sel]
                    body_mat = data.xmat[sel].reshape(3, 3)
                    world_pos = body_pos + body_mat @ local

                world_pos[2] = max(0.03, world_pos[2])
                data.mocap_pos[0] = world_pos

                ball_selected = False
                reaching = True
                model.geom_rgba[target_geom_id] = COLOR_MOVING
                print(f"  >> Going to ({world_pos[0]:+.3f}, {world_pos[1]:+.3f}, {world_pos[2]:+.3f})")

        # IK solve every frame while reaching
        if reaching:
            target_pos = data.mocap_pos[0].copy()

            for _ in range(IK_STEPS):
                mujoco.mj_forward(model, data)
                error = target_pos - data.site_xpos[ee_site_id]

                if np.linalg.norm(error) < 5e-4:
                    break

                err_norm = np.linalg.norm(error)
                if err_norm > MAX_DPOS:
                    error = error * MAX_DPOS / err_norm

                mujoco.mj_jacSite(model, data, jacp, None, ee_site_id)
                J = jacp[:, arm_joint_ids]
                JJT = J @ J.T + DAMPING * np.eye(3)
                dq = J.T @ np.linalg.solve(JJT, error)

                for i, jid in enumerate(arm_joint_ids):
                    data.qpos[jid] += dq[i]
                    lo, hi = model.jnt_range[jid]
                    data.qpos[jid] = np.clip(data.qpos[jid], lo, hi)

            for jid in arm_joint_ids:
                data.ctrl[jid] = data.qpos[jid]

            ee = data.site_xpos[ee_site_id]
            err = np.linalg.norm(ee - target_pos)
            if err < 0.01:
                reaching = False
                model.geom_rgba[target_geom_id] = COLOR_IDLE
                print(f"  >> Reached! (err: {err:.4f}m)")

        for _ in range(5):
            mujoco.mj_step(model, data)

        viewer.sync()

    viewer.close()

    print("\nFinal joint config:")
    angles = [f"{data.qpos[i]:+.4f}" for i in range(6)]
    print(f"  [{', '.join(angles)}]")


if __name__ == "__main__":
    main()
