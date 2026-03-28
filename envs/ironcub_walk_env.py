#!/usr/bin/env python3
"""
Gymnasium environment for iRonCub-Mk1_1 bipedal walking — PPO baseline.

Loads robot from model_stl.urdf via MjSpec (avoids model.xml kinematic bug).
Adds freejoint (floating base) + position actuators + floor plane programmatically.

Observation (63-dim):
    joint_pos (26) | joint_vel (26) | base_quat (4) | base_linvel (3) | base_angvel (3) | phase (1)

Action (26-dim):
    Normalized target joint positions in [-1, 1], mapped to joint limits.
    Position actuators apply PD control to reach targets.

Reward: reference motion tracking (RSI).
    Reset initializes from a random frame of the walking prior.
    Reward = exp(-2 * pose_err) + 0.1 * exp(-0.1 * vel_err) + small forward bonus.
    Phase (ref_frame / n_frames) is included in obs so the policy knows where it is
    in the gait cycle.
"""

import math
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

REPO_ROOT   = Path(__file__).resolve().parent.parent
URDF_PATH   = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf"
MESH_DIR    = REPO_ROOT / "assets/iRonCub/meshes/stl/"
MOTION_PATH = REPO_ROOT / "motion_priors/walking/07_12_retargeted_adherent.npy"

URDF_JOINT_ORDER = [
    "torso_pitch", "torso_roll", "torso_yaw",
    "neck_pitch",  "neck_roll",  "neck_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll",
]
N_JOINTS = len(URDF_JOINT_ORDER)   # 26

MIN_HEIGHT = 0.42     # m — below this = fallen
MAX_HEIGHT = 0.80     # m — above this = numerical explosion

# PD gains for position actuators
KP = 80.0    # Nm/rad
KV = 8.0     # Nm·s/rad

MAX_EPISODE_STEPS = 1000

# Motion is at 120fps, MuJoCo default timestep = 0.002s → advance ref every 4 sim steps
MOTION_FPS     = 120.0
SIM_TIMESTEP   = 0.002
REF_STRIDE     = max(1, round(1.0 / (MOTION_FPS * SIM_TIMESTEP)))   # = 4


def _build_model():
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

    for jname in URDF_JOINT_ORDER:
        act = spec.add_actuator()
        act.name       = f"{jname}_act"
        act.trntype    = mujoco.mjtTrn.mjTRN_JOINT
        act.target     = jname
        act.gaintype   = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = KP
        act.biastype   = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[1] = -KP
        act.biasprm[2] = -KV

    return spec.compile()


class IronCubWalkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = _build_model()
        self.data  = mujoco.MjData(self.model)
        self._viewer = None

        # Joint limits for action rescaling (skip freejoint at index 0)
        self.joint_lo = np.clip(self.model.jnt_range[1:, 0].copy(), -math.pi, 0.0)
        self.joint_hi = np.clip(self.model.jnt_range[1:, 1].copy(),  0.0, math.pi)

        # Motion prior
        motion = np.load(MOTION_PATH, allow_pickle=True).item()
        self._ref_joint_pos   = motion["joint_pos"].astype(np.float64)    # (N, 26)
        self._ref_joint_vel   = motion["joint_vel"].astype(np.float64)    # (N, 26)
        self._ref_root_pos    = motion["root_pos"].astype(np.float64)     # (N, 3)
        self._ref_root_quat   = motion["root_quat"].astype(np.float64)    # (N, 4) wxyz
        self._ref_root_linvel = motion["root_lin_vel"].astype(np.float64) # (N, 3)
        self._n_frames = len(self._ref_joint_pos)
        self._ref_frame = 0

        # obs: joint_pos(26) + joint_vel(26) + quat(4) + linvel(3) + angvel(3) + phase(1)
        obs_dim = N_JOINTS + N_JOINTS + 4 + 3 + 3 + 1   # 63
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32
        )

        self._step_count = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_obs(self):
        joint_pos   = self.data.qpos[7:].astype(np.float32)
        joint_vel   = self.data.qvel[6:].astype(np.float32)
        base_quat   = self.data.qpos[3:7].astype(np.float32)
        base_linvel = self.data.qvel[0:3].astype(np.float32)
        base_angvel = self.data.qvel[3:6].astype(np.float32)
        phase       = np.array([self._ref_frame / self._n_frames], dtype=np.float32)
        return np.concatenate([joint_pos, joint_vel, base_quat, base_linvel, base_angvel, phase])

    def _action_to_ctrl(self, action):
        return self.joint_lo + (action + 1.0) * 0.5 * (self.joint_hi - self.joint_lo)

    def _compute_reward(self, action):
        joint_pos   = self.data.qpos[7:].astype(np.float64)
        joint_vel   = self.data.qvel[6:].astype(np.float64)
        forward_vel = float(self.data.qvel[0])
        quat        = self.data.qpos[3:7]

        ref_jp = self._ref_joint_pos[self._ref_frame]
        ref_jv = self._ref_joint_vel[self._ref_frame]

        pose_err = float(np.sum(np.square(joint_pos - ref_jp)))
        vel_err  = float(np.sum(np.square(joint_vel - ref_jv)))

        # Core tracking signal — 1.0 when perfect, decays with error
        pose_reward = math.exp(-2.0 * pose_err)
        vel_reward  = math.exp(-0.1 * vel_err)

        # Upright gate — kills reward if robot tilts badly
        qx, qy = float(quat[1]), float(quat[2])
        upright = math.exp(-5.0 * (qx*qx + qy*qy))

        # Small forward bonus so it doesn't learn to track while drifting backward
        forward_bonus = max(0.0, forward_vel) * upright * 0.3

        ctrl_cost = -0.001 * float(np.sum(np.square(action)))

        return pose_reward + 0.1 * vel_reward + forward_bonus + ctrl_cost

    def _is_terminated(self):
        h = float(self.data.qpos[2])
        return h < MIN_HEIGHT or h > MAX_HEIGHT

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # RSI: start from a random frame in the motion prior
        fi = int(self.np_random.integers(0, self._n_frames))
        self._ref_frame = fi

        self.data.qpos[0]   = 0.0
        self.data.qpos[1]   = 0.0
        self.data.qpos[2]   = float(self._ref_root_pos[fi, 2])
        self.data.qpos[3:7] = self._ref_root_quat[fi]
        self.data.qpos[7:]  = self._ref_joint_pos[fi]
        self.data.qvel[0:3] = self._ref_root_linvel[fi]
        self.data.qvel[6:]  = self._ref_joint_vel[fi]

        # Small noise to break symmetry across parallel envs
        self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02, N_JOINTS)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = self._action_to_ctrl(action)

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        # Advance reference at the motion's natural rate
        if self._step_count % REF_STRIDE == 0:
            self._ref_frame = (self._ref_frame + 1) % self._n_frames

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated() or bool(np.any(~np.isfinite(obs)))
        truncated  = self._step_count >= MAX_EPISODE_STEPS

        if terminated and not np.all(np.isfinite(obs)):
            obs = np.zeros_like(obs)

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
