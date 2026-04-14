#!/usr/bin/env python3
"""
Retarget CMU BVH mocap to iRonCub joint angles — direct Euler mapping (no IK).

BVH→world: Z_bvh→X_world (forward), X_bvh→Y_world (lateral), Y_bvh→Z_world (up)
Hip/knee sign: BVH Xrot > 0 = flex forward → robot hip_pitch/knee negative (axis [0,-1,0])
Ankle: l_ankle_pitch axis [0,+1,0] = same-direction as BVH Xrot → same sign

Usage:
    python scripts/retarget_bvh_smoothed.py <bvh> <output.npy> [config.yaml]
"""

import sys, math
import numpy as np
import yaml
import idyntree.bindings as idyn
from bvh import Bvh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as SciRot

URDF_JOINT_ORDER = [
    "torso_pitch", "torso_roll", "torso_yaw",
    "neck_pitch",  "neck_roll",  "neck_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll",
]
L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW = 14, 15, 16
L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL = 17, 18, 19
R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW = 20, 21, 22
R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL = 23, 24, 25

# BVH (Y-up, +Z forward) → robot world (Z-up, +X forward)
BVH2W     = np.array([[0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.]])
BVH2W_inv = BVH2W.T
F_MAT = R_W = BVH2W
R_W_inv    = BVH2W_inv
D2R = math.pi / 180.0


def _bvh_scale(bvh):
    off = np.linalg.norm(bvh.joint_offset('LeftLeg'))
    s = 0.42 / off
    print(f"  BVH femur: {off:.3f} units  scale: {s:.5f} m/unit")
    return s


def _robot_leg_len(model):
    kd = idyn.KinDynComputations()
    kd.loadRobotModel(model)
    n = model.getNrOfDOFs()
    q0 = idyn.VectorDynSize(n); q0.zero()
    dq0 = idyn.VectorDynSize(n); dq0.zero()
    g = idyn.Vector3(); g.setVal(2, -9.81)
    kd.setRobotState(idyn.Transform.Identity(), q0, idyn.Twist(), dq0, g)
    fi = model.getFrameIndex('l_sole_1')
    if fi < 0:
        return 0.63
    z = float(kd.getWorldTransform(fi).getPosition().getVal(2))
    print(f"  Robot leg length: {abs(z):.4f} m")
    return abs(z)


def _joint_limits(model):
    lb, ub = [], []
    for name in URDF_JOINT_ORDER:
        ji = model.getJointIndex(name)
        j  = model.getJoint(ji)
        try:
            lo = max(float(j.getMinPosLimit(0)), -math.pi)
            hi = min(float(j.getMaxPosLimit(0)),  math.pi)
        except Exception:
            lo, hi = -math.pi, math.pi
        lb.append(lo); ub.append(hi)
    return np.array(lb), np.array(ub)


def _bvh_euler(bvh, joint, fi):
    """(Xrot, Yrot, Zrot) in degrees for a BVH joint at frame fi."""
    try:
        rx = float(bvh.frame_joint_channel(fi, joint, 'Xrotation'))
        ry = float(bvh.frame_joint_channel(fi, joint, 'Yrotation'))
        rz = float(bvh.frame_joint_channel(fi, joint, 'Zrotation'))
        return rx, ry, rz
    except Exception:
        return 0.0, 0.0, 0.0


def _bvh_local_rot(bvh, joint_name, frame):
    rz = float(bvh.frame_joint_channel(frame, joint_name, 'Zrotation'))
    ry = float(bvh.frame_joint_channel(frame, joint_name, 'Yrotation'))
    rx = float(bvh.frame_joint_channel(frame, joint_name, 'Xrotation'))
    return SciRot.from_euler('ZYX', [rz, ry, rx], degrees=True).as_matrix()


def _smooth(arr, window=15, poly=3):
    if arr.shape[0] <= window:
        return arr.copy()
    out = np.empty_like(arr)
    for i in range(arr.shape[1]):
        out[:, i] = savgol_filter(arr[:, i], window, poly)
    return out


def retarget_bvh_to_robot(bvh_path, urdf_path, output_path):

    print(f"\nLoading BVH: {bvh_path}")
    with open(bvh_path) as f:
        bvh = Bvh(f.read())
    n_frames = bvh.nframes
    dt       = bvh.frame_time
    fps      = 1.0 / dt
    print(f"  Frames: {n_frames}  FPS: {fps:.1f}")
    scale = _bvh_scale(bvh)

    print(f"\nLoading URDF: {urdf_path}")
    loader = idyn.ModelLoader()
    if not loader.loadReducedModelFromFile(urdf_path, URDF_JOINT_ORDER):
        raise RuntimeError("Failed to load URDF")
    model = loader.model()
    dofs  = model.getNrOfDOFs()
    assert dofs == len(URDF_JOINT_ORDER), f"Expected {len(URDF_JOINT_ORDER)} DOFs, got {dofs}"

    robot_leg = _robot_leg_len(model)
    all_hip_y = [float(bvh.frame_joint_channel(f, 'Hips', 'Yposition'))
                 for f in range(n_frames)]
    bvh_hip_h = float(np.median(all_hip_y)) * scale
    leg_scale = robot_leg / max(bvh_hip_h, 0.1) * 0.9
    print(f"  BVH hip height (median): {bvh_hip_h:.4f} m  leg_scale: {leg_scale:.4f}")

    lb_arr, ub_arr = _joint_limits(model)

    # Skip leading T-pose frames (Hips position = 0)
    start_frame = 0
    for fi in range(n_frames):
        px = abs(float(bvh.frame_joint_channel(fi, 'Hips', 'Xposition')))
        py = abs(float(bvh.frame_joint_channel(fi, 'Hips', 'Yposition')))
        pz = abs(float(bvh.frame_joint_channel(fi, 'Hips', 'Zposition')))
        if px + py + pz > 1.0:
            start_frame = fi
            break
    print(f"  Walking starts at frame {start_frame}.")

    joint_pos = np.zeros((n_frames, dofs), dtype=np.float32)
    root_pos  = np.zeros((n_frames, 3),    dtype=np.float32)
    root_quat = np.zeros((n_frames, 4),    dtype=np.float32)  # wxyz

    for fi in range(start_frame, n_frames):
        if (fi - start_frame) % 50 == 0:
            print(f"  Frame {fi}/{n_frames}")

        # ── Root: world position + yaw-only orientation ──────────────────────
        hips_pos_bvh = np.array([
            float(bvh.frame_joint_channel(fi, 'Hips', 'Xposition')),
            float(bvh.frame_joint_channel(fi, 'Hips', 'Yposition')),
            float(bvh.frame_joint_channel(fi, 'Hips', 'Zposition')),
        ])
        hips_rot_bvh = _bvh_local_rot(bvh, 'Hips', fi)
        hips_pos_w = F_MAT @ hips_pos_bvh * scale * leg_scale
        hips_rot_w = R_W @ hips_rot_bvh @ R_W_inv
        yaw = math.atan2(hips_rot_w[1, 0], hips_rot_w[0, 0]) + math.pi
        hips_rot_w = np.array([[math.cos(yaw), -math.sin(yaw), 0.],
                                [math.sin(yaw),  math.cos(yaw), 0.],
                                [0.,             0.,            1.]])
        root_pos[fi] = hips_pos_w
        q_xyzw = SciRot.from_matrix(hips_rot_w).as_quat()
        root_quat[fi] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

        # ── Legs: direct BVH Euler → robot joint angles ──────────────────────
        # Axis analysis (from URDF):
        #   hip_pitch axis [0,-1,0] in world → positive = backward.
        #     BVH UpLeg Xrot > 0 = thigh forward → hip_pitch = -Xrot
        #   knee      axis [0,-1,0] in world → positive = extension.
        #     BVH Leg Xrot > 0 = knee flex   → knee      = -Xrot
        #   ankle_pitch axis [0,+1,0] in world → positive = dorsiflexion.
        #     BVH Foot Xrot maps to same +Y rotation → ankle_pitch = +Xrot
        q = np.zeros(dofs)

        for prefix, hip_p, hip_r, hip_y, kn, ank_p, ank_r in [
            ('Left',  L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL),
            ('Right', R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL),
        ]:
            upleg_rx, _, _ = _bvh_euler(bvh, f'{prefix}UpLeg', fi)
            leg_rx,   _, _ = _bvh_euler(bvh, f'{prefix}Leg',   fi)
            foot_rx,  _, _ = _bvh_euler(bvh, f'{prefix}Foot',  fi)

            q[hip_p] = -upleg_rx * D2R   # forward flex → negative
            q[hip_r] = 0.0
            q[hip_y] = 0.0
            q[kn]    = -leg_rx   * D2R   # knee flex → negative
            q[ank_p] =  foot_rx  * D2R   # ankle: same-sign mapping
            q[ank_r] = 0.0

        joint_pos[fi] = np.clip(q, lb_arr, ub_arr)

    # Fill T-pose frames
    if start_frame > 0:
        joint_pos[:start_frame] = joint_pos[start_frame]
        root_pos[:start_frame]  = root_pos[start_frame]
        root_quat[:start_frame] = root_quat[start_frame]

    # Smooth
    print("\nSmoothing trajectories...")
    joint_pos = np.clip(_smooth(joint_pos, window=15), lb_arr, ub_arr).astype(np.float32)
    root_pos  = _smooth(root_pos, window=15).astype(np.float32)

    # Velocities
    JOINT_VEL_LIMS = np.array([
        10., 10., 10., 10., 10., 10.,
        10., 10., 10., 10., 10., 10., 10., 10.,
        5.1, 7.64, 7.64, 7.64, 7.64, 7.64,
        5.1, 7.64, 7.64, 7.64, 7.64, 7.64,
    ], dtype=np.float32)

    joint_vel    = np.zeros_like(joint_pos)
    root_lin_vel = np.zeros_like(root_pos)
    joint_vel[:-1]    = np.diff(joint_pos, axis=0) / dt
    root_lin_vel[:-1] = np.diff(root_pos,  axis=0) / dt
    joint_vel[-1]    = joint_vel[-2]
    root_lin_vel[-1] = root_lin_vel[-2]
    joint_vel    = np.clip(_smooth(joint_vel,    window=15), -JOINT_VEL_LIMS, JOINT_VEL_LIMS)
    root_lin_vel = np.clip(_smooth(root_lin_vel, window=15), -2.0, 2.0)

    root_ang_vel = np.zeros((n_frames, 3), dtype=np.float32)
    for i in range(n_frames - 1):
        q0 = SciRot.from_quat([root_quat[i,1],   root_quat[i,2],   root_quat[i,3],   root_quat[i,0]])
        q1 = SciRot.from_quat([root_quat[i+1,1], root_quat[i+1,2], root_quat[i+1,3], root_quat[i+1,0]])
        root_ang_vel[i] = (q0.inv() * q1).as_rotvec() / dt
    root_ang_vel[-1] = root_ang_vel[-2]
    root_ang_vel = np.clip(_smooth(root_ang_vel, window=15), -3.0, 3.0)

    result = {
        "root_pos":     root_pos.astype(np.float32),
        "root_quat":    root_quat.astype(np.float32),
        "joint_pos":    joint_pos.astype(np.float32),
        "root_lin_vel": root_lin_vel.astype(np.float32),
        "root_ang_vel": root_ang_vel.astype(np.float32),
        "joint_vel":    joint_vel.astype(np.float32),
        "fps":          np.float32(fps),
        "joint_names":  np.array(URDF_JOINT_ORDER),
    }
    np.save(output_path, result)
    print(f"\nSaved: {output_path}")
    print(f"  root_pos z:    [{root_pos[:,2].min():.3f}, {root_pos[:,2].max():.3f}] m")
    print(f"  l_hip_pitch:   [{joint_pos[:,L_HIP_PITCH].min():.3f}, {joint_pos[:,L_HIP_PITCH].max():.3f}] rad")
    print(f"  l_knee:        [{joint_pos[:,L_KNEE].min():.3f}, {joint_pos[:,L_KNEE].max():.3f}] rad")
    print(f"  l_ankle_pitch: [{joint_pos[:,L_ANKLE_PITCH].min():.3f}, {joint_pos[:,L_ANKLE_PITCH].max():.3f}] rad")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: retarget_bvh_smoothed.py <bvh> <output.npy> [config.yaml]")
        sys.exit(1)

    bvh_file    = sys.argv[1]
    output_file = sys.argv[2]
    config_file = sys.argv[3] if len(sys.argv) > 3 else "configs/ironcub_bvh_mapping.yaml"

    with open(config_file) as f:
        cfg = yaml.safe_load(f)

    retarget_bvh_to_robot(bvh_file, cfg["urdf_path"], output_file)
