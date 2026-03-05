"""Visualize the Piper robotic arm in Isaac Lab (requires Isaac Sim + Isaac Lab installed)."""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Piper Arm Visualization in Isaac Lab")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher (Isaac Sim is now running) ---
import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.sim import SimulationContext

PIPER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Piper",
    spawn=sim_utils.UrdfFileCfg(
        asset_path="piper_description/urdf/piper_description.urdf",
        fix_base=True,
        make_instanceable=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.5,
            "joint3": -1.0,
            "joint4": 0.0,
            "joint5": 0.5,
            "joint6": 0.0,
            "joint7": 0.02,
            "joint8": -0.02,
        },
    ),
)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])

    # Ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # Dome light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # Spawn Piper arm
    piper = Articulation(PIPER_CFG)

    sim.reset()
    print("[INFO] Piper arm spawned. Running simulation...")
    print("[INFO] Joint names:", piper.joint_names)

    # Simple sinusoidal joint motion for visualization
    step = 0
    while simulation_app.is_running():
        if step % 200 == 0:
            # Reset to initial pose periodically
            piper.write_joint_state_to_sim(
                piper.data.default_joint_pos, piper.data.default_joint_vel
            )
            piper.reset()

        t = step * sim_cfg.dt
        # Gentle sinusoidal motion on joints 1, 2, 5
        targets = piper.data.default_joint_pos.clone()
        targets[:, 0] += 0.5 * torch.sin(torch.tensor(t * 1.0))  # joint1
        targets[:, 1] += 0.3 * torch.sin(torch.tensor(t * 0.8))  # joint2
        targets[:, 4] += 0.3 * torch.sin(torch.tensor(t * 1.2))  # joint5
        piper.set_joint_position_target(targets)
        piper.write_data_to_sim()

        sim.step()
        step += 1
        piper.update(sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
