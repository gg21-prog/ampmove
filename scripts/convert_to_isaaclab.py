#!/usr/bin/env python3
"""
Convert retargeted .npy motion prior to Isaac Lab AMP format (.npz).

Takes the output of retarget_bvh_smoothed.py and adds:
  - key_body_pos:   (N, K, 3)  root-relative world positions of key bodies
  - key_body_rot6d: (N, K, 6)  6D rotation of key bodies (first 2 cols of rot matrix)
  - contact:        (N, 2)     foot contact flags [left, right]

Key bodies (in order): l_sole_1, r_sole_1

NOTE: verify KEY_BODY_FRAMES matches what your Isaac Lab AMP task config expects.
NOTE: verify URDF_JOINT_ORDER matches Isaac Lab asset dof_names order before training.

Usage:
    python scripts/convert_to_isaaclab.py <input.npy> <output.npz>
"""

import sys
import numpy as np
import idyntree.bindings as idyn
from pathlib import Path
from scipy.spatial.transform import Rotation as SciRot

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/model.urdf"

URDF_JOINT_ORDER = [
    "torso_pitch", "torso_roll", "torso_yaw",
    "neck_pitch",  "neck_roll",  "neck_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll",
]

# Key body frames for AMP discriminator observation — extend if needed
KEY_BODY_FRAMES = ["l_sole_1", "r_sole_1"]
CONTACT_FRAMES  = ["l_sole_1", "r_sole_1"]   # same order: [left, right]
CONTACT_THRESH  = 0.02   # m — sole z below this = foot contact


def rot_to_6d(R):
    """3x3 rotation matrix → 6D (first two columns, row-major flattened)."""
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def quat_wxyz_to_rot(q):
    """wxyz quaternion → 3x3 rotation matrix."""
    w, x, y, z = q
    return SciRot.from_quat([x, y, z, w]).as_matrix()


def make_idyn_transform(pos_np, rot_np):
    rot = idyn.Rotation()
    for r in range(3):
        for c in range(3):
            rot.setVal(r, c, float(rot_np[r, c]))
    pos = idyn.Position()
    for i in range(3):
        pos.setVal(i, float(pos_np[i]))
    return idyn.Transform(rot, pos)


def main():
    if len(sys.argv) < 3:
        print("Usage: convert_to_isaaclab.py <input.npy> <output.npz>")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    # ── Load retargeted motion ────────────────────────────────────────────────
    print(f"Loading: {input_path}")
    d = np.load(input_path, allow_pickle=True).item()
    root_pos     = d["root_pos"]       # (N, 3)
    root_quat    = d["root_quat"]      # (N, 4) wxyz
    joint_pos    = d["joint_pos"]      # (N, 26)
    joint_vel    = d["joint_vel"]      # (N, 26)
    root_lin_vel = d["root_lin_vel"]   # (N, 3)
    root_ang_vel = d["root_ang_vel"]   # (N, 3)
    fps          = float(d["fps"])
    dt           = 1.0 / fps
    N            = root_pos.shape[0]
    print(f"  {N} frames @ {fps:.1f} fps  ({N/fps:.2f}s)")

    # ── Load URDF for FK ──────────────────────────────────────────────────────
    print(f"\nLoading URDF: {URDF_PATH}")
    loader = idyn.ModelLoader()
    if not loader.loadReducedModelFromFile(str(URDF_PATH), URDF_JOINT_ORDER):
        raise RuntimeError("Failed to load URDF")
    model = loader.model()
    kd    = idyn.KinDynComputations()
    kd.loadRobotModel(model)
    n_dof = model.getNrOfDOFs()
    assert n_dof == len(URDF_JOINT_ORDER), f"DOF mismatch: {n_dof} vs {len(URDF_JOINT_ORDER)}"

    gravity = idyn.Vector3()
    gravity.setVal(2, -9.81)

    # Validate all frame names up front
    all_frames = list(dict.fromkeys(KEY_BODY_FRAMES + CONTACT_FRAMES))
    frame_indices = {}
    for name in all_frames:
        fi = model.getFrameIndex(name)
        if fi < 0:
            raise RuntimeError(f"Frame '{name}' not found in URDF — check KEY_BODY_FRAMES")
        frame_indices[name] = fi
    print(f"  Key body frames: {KEY_BODY_FRAMES}")

    # ── FK pass ───────────────────────────────────────────────────────────────
    n_key          = len(KEY_BODY_FRAMES)
    key_body_pos   = np.zeros((N, n_key, 3), dtype=np.float32)
    key_body_rot6d = np.zeros((N, n_key, 6), dtype=np.float32)
    contact        = np.zeros((N, 2),        dtype=np.float32)

    q_idyn  = idyn.VectorDynSize(n_dof)
    dq_idyn = idyn.VectorDynSize(n_dof)
    dq_idyn.zero()

    print("\nRunning forward kinematics...")
    for fi in range(N):
        if fi % 50 == 0:
            print(f"  Frame {fi}/{N}")

        root_R   = quat_wxyz_to_rot(root_quat[fi])
        root_p   = root_pos[fi]
        T_root   = make_idyn_transform(root_p, root_R)
        root_R_T = root_R.T   # precompute inverse (rotation transpose = inverse)

        for j in range(n_dof):
            q_idyn.setVal(j, float(joint_pos[fi, j]))

        kd.setRobotState(T_root, q_idyn, idyn.Twist(), dq_idyn, gravity)

        # Key body states (root-relative)
        for ki, frame_name in enumerate(KEY_BODY_FRAMES):
            T   = kd.getWorldTransform(frame_indices[frame_name])
            p_w = np.array([T.getPosition().getVal(i) for i in range(3)])
            R_w = np.array([[T.getRotation().getVal(r, c) for c in range(3)] for r in range(3)])

            key_body_pos[fi, ki]   = (root_R_T @ (p_w - root_p)).astype(np.float32)
            key_body_rot6d[fi, ki] = rot_to_6d(root_R_T @ R_w)

        # Contact flags
        for ci, frame_name in enumerate(CONTACT_FRAMES):
            T = kd.getWorldTransform(frame_indices[frame_name])
            contact[fi, ci] = 1.0 if T.getPosition().getVal(2) < CONTACT_THRESH else 0.0

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving: {output_path}")
    np.savez(
        output_path,
        # Root state
        root_pos     = root_pos,
        root_rot     = root_quat,        # wxyz — rename to match Isaac Lab if needed
        root_vel     = root_lin_vel,
        root_ang_vel = root_ang_vel,
        # Joint state
        dof_pos      = joint_pos,
        dof_vel      = joint_vel,
        # Key body state (root-relative)
        key_body_pos   = key_body_pos,
        key_body_rot6d = key_body_rot6d,
        # Contact
        contact = contact,
        # Metadata
        fps            = np.float32(fps),
        dt             = np.float32(dt),
        joint_names    = np.array(URDF_JOINT_ORDER),
        key_body_names = np.array(KEY_BODY_FRAMES),
    )

    print("Done.")
    print(f"  dof_pos shape:        {joint_pos.shape}")
    print(f"  key_body_pos shape:   {key_body_pos.shape}")
    print(f"  key_body_rot6d shape: {key_body_rot6d.shape}")
    print(f"  contact shape:        {contact.shape}")
    print(f"  left  foot contact:   {contact[:, 0].mean()*100:.1f}% of frames")
    print(f"  right foot contact:   {contact[:, 1].mean()*100:.1f}% of frames")
    print()
    print("IMPORTANT: before training, verify:")
    print("  1. joint_names order matches Isaac Lab asset dof_names")
    print("  2. key_body_names match what the AMP task observation builder expects")
    print("  3. root_rot quaternion convention (wxyz here) matches Isaac Lab (usually xyzw)")


if __name__ == "__main__":
    main()
