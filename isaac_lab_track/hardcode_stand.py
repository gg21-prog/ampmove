#!/usr/bin/env python3
"""
Hardcoded stand test — spawn iRonCub in Isaac Lab and hold it at a stable
standing pose via PD control.

Run this before training to verify:
  - USD asset loads and all joints resolve correctly
  - PD gains in ironcub_cfg.py keep the robot balanced
  - Base height settles near the expected ~0.57 m

Usage:
    python isaac_lab_track/hardcode_stand.py
    python isaac_lab_track/hardcode_stand.py --headless
    python isaac_lab_track/hardcode_stand.py --duration 20.0
"""

import argparse
import sys
from pathlib import Path

# AppLauncher must be created before any omni / Isaac imports
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--duration", type=float, default=10.0)
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

REPO_ROOT  = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PHYSICS_DT = 1.0 / 120.0
RENDER_DT  = 1.0 / 60.0

# Slight knee bend + matching hip/ankle to keep COM over feet.
# All other joints default to 0.
STAND_POSE = {
    "l_hip_pitch":   -0.06,
    "r_hip_pitch":   -0.06,
    "l_knee":         0.12,
    "r_knee":         0.12,
    "l_ankle_pitch":  0.06,
    "r_ankle_pitch":  0.06,
    # Elbows have limits [0.262, 1.850] rad; 0.0 is out of range
    "l_elbow":        0.5,
    "r_elbow":        0.5,
}


def build_scene() -> Articulation:
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_light(
        "/World/light",
        sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )
    from isaac_lab_track.ironcub_cfg import IRONCUB_CFG
    return Articulation(IRONCUB_CFG.replace(prim_path="/World/iRonCub"))


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=max(1, round(RENDER_DT / PHYSICS_DT)),
    ))
    sim.set_camera_view(eye=(2.5, 2.0, 1.2), target=(0.0, 0.0, 0.6))

    robot = build_scene()
    sim.reset()
    robot.update(PHYSICS_DT)

    n_dof     = robot.num_joints
    dof_names = list(robot.joint_names)

    # Build target tensor from STAND_POSE dict
    target = torch.zeros(1, n_dof)
    for name, angle in STAND_POSE.items():
        matches = [i for i, n in enumerate(dof_names) if n == name]
        if not matches:
            print(f"  WARNING: joint '{name}' not found in asset — skipping")
        else:
            target[0, matches[0]] = angle

    n_steps      = int(args.duration / PHYSICS_DT)
    report_every = max(1, int(1.0 / PHYSICS_DT))

    print(f"\niRonCub hardcode-stand  |  {args.duration:.1f}s  |  {n_steps} steps")
    print(f"DOFs ({n_dof}): {dof_names}")
    print("Press Ctrl+C to exit early.\n")

    for step in range(n_steps):
        if not simulation_app.is_running():
            break

        robot.set_joint_position_target(target)
        sim.step()
        robot.update(PHYSICS_DT)

        if step % report_every == 0:
            h = float(robot.data.root_pos_w[0, 2])
            t = step * PHYSICS_DT
            print(f"  t={t:5.1f}s  base_height={h:.4f}m")

    simulation_app.close()


if __name__ == "__main__":
    main()
