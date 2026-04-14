#!/usr/bin/env python3
"""
Visualize retargeted motion on the iRonCub MuJoCo model.

Usage:
    python scripts/visualize_retargeted.py [npy_file] [--speed 1.0]

Default npy_file: motion_priors/walking/07_12_retargeted_adherent.npy
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT  = Path(__file__).resolve().parent.parent
URDF_PATH  = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf"
MESH_DIR   = REPO_ROOT / "assets/iRonCub/meshes/stl/"
DEFAULT_NPY = REPO_ROOT / "motion_priors/walking/07_12_retargeted_adherent.npy"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npy_file", nargs="?", default=str(DEFAULT_NPY))
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (default 1.0)")
    args = parser.parse_args()

    # Load motion data
    print(f"Loading motion: {args.npy_file}")
    data_dict = np.load(args.npy_file, allow_pickle=True).item()
    root_pos  = data_dict["root_pos"]   # (N, 3)
    root_quat = data_dict["root_quat"]  # (N, 4) wxyz
    joint_pos = data_dict["joint_pos"]  # (N, 26)
    fps       = float(data_dict["fps"])
    n_frames  = root_pos.shape[0]
    dt        = 1.0 / fps
    print(f"  Frames: {n_frames}  FPS: {fps}  Duration: {n_frames/fps:.2f}s")

    # Load MuJoCo model from URDF (avoids body-orientation bugs in pre-generated model.xml)
    print(f"Loading model: {URDF_PATH}")
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    spec.meshdir = str(MESH_DIR)
    # URDF has no floating base — add freejoint to root body so robot can translate/rotate
    fj = spec.worldbody.first_body().add_joint()
    fj.type = mujoco.mjtJoint.mjJNT_FREE
    fj.name = "root_freejoint"
    model = spec.compile()
    data  = mujoco.MjData(model)

    # Sanity check: model should have freejoint (7) + 26 joints = 33 qpos
    expected_qpos = 7 + joint_pos.shape[1]
    if model.nq != expected_qpos:
        print(f"Warning: model.nq={model.nq}, expected {expected_qpos}. "
              f"Joint count mismatch — playback may be wrong.")

    print(f"\nPlaying back at {args.speed}x speed. Close viewer to exit.")
    print("Press R to restart loop.")

    frame = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t_start = time.time()

            # Set robot state from motion data
            # qpos layout: [x, y, z, qw, qx, qy, qz, joint_0 ... joint_25]
            data.qpos[0:3] = root_pos[frame]
            data.qpos[3:7] = root_quat[frame]   # wxyz — matches MuJoCo freejoint order
            data.qpos[7:]  = joint_pos[frame]
            data.qvel[:]   = 0.0                 # zero velocity for clean display

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame = (frame + 1) % n_frames
            if frame == 0:
                print("  Looping...")

            # Real-time pacing
            elapsed = time.time() - t_start
            sleep_t = (dt / args.speed) - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
