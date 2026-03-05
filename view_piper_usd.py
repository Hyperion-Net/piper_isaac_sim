"""Visualize the Piper arm using the pre-built USD file directly in Isaac Sim standalone."""

import argparse

from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Piper Arm USD Viewer")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Imports after AppLauncher ---
import os
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext
from omni.usd import get_context
from pxr import Sdf, UsdGeom

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.join(SCRIPT_DIR, "USD", "piper_v1.usd")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])

    # Ground plane + light
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    # Add the pre-built USD as a reference
    stage = get_context().get_stage()
    prim = stage.DefinePrim("/World/Piper", "Xform")
    prim.GetReferences().AddReference(USD_PATH)

    sim.reset()
    print(f"[INFO] Loaded Piper USD from: {USD_PATH}")
    print("[INFO] Simulation running — close the window to exit.")

    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    main()
    simulation_app.close()
