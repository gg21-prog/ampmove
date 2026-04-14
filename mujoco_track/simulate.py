#!/usr/bin/env python3
"""
Simulate iRonCub directly from URDF in MuJoCo — no gym wrapper.

Usage:
    python mujoco_track/simulate.py              # zero actions (hold position)
    python mujoco_track/simulate.py --random     # random actions
"""

import argparse
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf"
MESH_DIR  = REPO_ROOT / "assets/iRonCub/meshes/stl/"


def build_model():
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    spec.meshdir = str(MESH_DIR)

    fj = spec.worldbody.first_body().add_joint()
    fj.type = mujoco.mjtJoint.mjJNT_FREE
    fj.name = "root_freejoint"

    floor = spec.worldbody.add_geom()
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [20.0, 20.0, 0.01]
    floor.pos  = [0.0, 0.0, 0.0]
    floor.name = "floor"

    return spec.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true",
                        help="Apply small random actions instead of zero")
    args = parser.parse_args()

    model = build_model()
    data  = mujoco.MjData(model)

    print(f"iRonCub loaded — nq={model.nq}  nv={model.nv}  nu={model.nu}")

    # Start at standing height
    data.qpos[2] = 0.57
    data.qpos[3] = 1.0   # qw — upright orientation
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.random and model.nu > 0:
                data.ctrl[:] = np.random.uniform(-0.05, 0.05, model.nu)

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
