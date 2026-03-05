"""Lightweight Piper arm viewer using PyBullet (works without Isaac Sim / NVIDIA GPU)."""

import os
import time
import math
import pybullet as p
import pybullet_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "piper_description", "urdf", "piper_description.urdf")


def main():
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(SCRIPT_DIR)
    p.setGravity(0, 0, -9.81)

    # Ground plane
    p.loadURDF("plane.urdf")

    # Load Piper arm
    piper_id = p.loadURDF(
        URDF_PATH,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )

    num_joints = p.getNumJoints(piper_id)
    print(f"[INFO] Piper loaded with {num_joints} joints:")
    joint_indices = []
    for i in range(num_joints):
        info = p.getJointInfo(piper_id, i)
        name = info[1].decode("utf-8")
        joint_type = info[2]
        lower, upper = info[8], info[9]
        print(f"  [{i}] {name}  type={joint_type}  limits=[{lower:.3f}, {upper:.3f}]")
        if joint_type != p.JOINT_FIXED:
            joint_indices.append(i)

    # Camera setup
    p.resetDebugVisualizerCamera(
        cameraDistance=0.8,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0, 0, 0.3],
    )

    print("[INFO] Running sinusoidal joint demo. Close the window to exit.")
    t = 0.0
    dt = 1 / 240.0
    p.setTimeStep(dt)

    try:
        while True:
            # Animate arm joints with sinusoidal motion
            for idx in joint_indices:
                info = p.getJointInfo(piper_id, idx)
                lower, upper = info[8], info[9]
                mid = (lower + upper) / 2
                amp = (upper - lower) / 4
                freq = 0.3 + 0.1 * idx
                target = mid + amp * math.sin(2 * math.pi * freq * t)
                p.setJointMotorControl2(
                    piper_id, idx, p.POSITION_CONTROL, targetPosition=target
                )

            p.stepSimulation()
            t += dt
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        p.disconnect()


if __name__ == "__main__":
    main()
