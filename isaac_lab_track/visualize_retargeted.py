#!/usr/bin/env python3
"""
Simulate iRonCub in Isaac Lab, replaying the retargeted walking motion prior.

Prerequisites — run once on the target machine from repo root:
    python {ISAACLAB}/scripts/tools/convert_urdf.py \
        assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf \
        assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd \
        --merge-fixed-joints

Usage:
    python scripts/simulate_isaaclab.py
    python scripts/simulate_isaaclab.py --headless
"""

import argparse
import sys
from pathlib import Path

# AppLauncher must be created before any omni/Isaac imports
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

REPO_ROOT   = Path(__file__).resolve().parent.parent
MOTION_PATH = REPO_ROOT / "motion_priors/walking/07_12_retargeted_adherent.npy"

# Motion is at 120fps — run physics at the same rate for clean 1:1 replay
PHYSICS_DT = 1.0 / 120.0
RENDER_DT  = 1.0 / 60.0   # render every 2 physics steps


def build_scene():
    sim_utils.spawn_ground_plane("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.spawn_distant_light(
        "/World/light",
        sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
    )

    # Import here (after app launch) to avoid premature omni initialisation
    sys.path.insert(0, str(REPO_ROOT))
    from isaac_lab_track.ironcub_cfg import IRONCUB_CFG

    robot = Articulation(IRONCUB_CFG.replace(prim_path="/World/iRonCub"))
    return robot


def main():
    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            render_interval=max(1, round(RENDER_DT / PHYSICS_DT)),
        )
    )
    sim.set_camera_view(eye=(2.5, 2.5, 1.5), target=(0.0, 0.0, 0.8))

    robot = build_scene()
    sim.reset()
    robot.update(PHYSICS_DT)

    # Load motion prior
    motion     = np.load(MOTION_PATH, allow_pickle=True).item()
    ref_jp     = torch.tensor(motion["joint_pos"],    dtype=torch.float32)  # (N, 26)
    ref_jv     = torch.tensor(motion["joint_vel"],    dtype=torch.float32)  # (N, 26)
    ref_pos    = torch.tensor(motion["root_pos"],     dtype=torch.float32)  # (N, 3)
    ref_quat   = torch.tensor(motion["root_quat"],    dtype=torch.float32)  # (N, 4) wxyz
    ref_linvel = torch.tensor(motion["root_lin_vel"], dtype=torch.float32)  # (N, 3)
    n_frames   = ref_jp.shape[0]

    # Build index mapping: motion joint order → Isaac Lab DOF order
    motion_names = list(motion["joint_names"])
    dof_names    = robot.joint_names
    try:
        dof_idx = [motion_names.index(n) for n in dof_names]
    except ValueError as e:
        raise RuntimeError(
            f"Joint name mismatch between motion prior and USD asset.\n"
            f"Motion joints: {motion_names}\nAsset joints:  {dof_names}"
        ) from e

    print(f"Replaying {n_frames} frames at {1/PHYSICS_DT:.0f} fps. Ctrl+C to quit.")

    frame = 0
    while simulation_app.is_running():
        fi = frame % n_frames

        # Root state: [pos(3), quat_wxyz(4), linvel(3), angvel(3)] — shape (1, 13)
        root_state = robot.data.default_root_state.clone()
        root_state[0, 0:3] = ref_pos[fi]
        root_state[0, 3:7] = ref_quat[fi]     # wxyz — matches Isaac Lab convention
        root_state[0, 7:10] = ref_linvel[fi]
        robot.write_root_state_to_sim(root_state)

        # Joint state: positions and velocities reordered to asset DOF order
        jp = ref_jp[fi, dof_idx].unsqueeze(0)   # (1, n_dof)
        jv = ref_jv[fi, dof_idx].unsqueeze(0)
        robot.write_joint_state_to_sim(jp, jv)

        sim.step()
        robot.update(PHYSICS_DT)
        frame += 1

    simulation_app.close()


if __name__ == "__main__":
    main()
