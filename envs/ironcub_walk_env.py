#!/usr/bin/env python3
"""
Gymnasium environment for iRonCub-Mk1_1 bipedal walking — PPO baseline.

Loads robot from model_stl.urdf via MjSpec (avoids model.xml kinematic bug).
Adds freejoint (floating base) + position actuators + floor plane programmatically.

Observation (62-dim):
    joint_pos (26) | joint_vel (26) | base_quat (4) | base_linvel (3) | base_angvel (3)

Action (26-dim):
    Normalized target joint positions in [-1, 1], mapped to joint limits.
    Position actuators apply PD control to reach targets.
"""

import math
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf"
MESH_DIR  = REPO_ROOT / "assets/iRonCub/meshes/stl/"

URDF_JOINT_ORDER = [
    "torso_pitch", "torso_roll", "torso_yaw",
    "neck_pitch",  "neck_roll",  "neck_yaw",
    "r_shoulder_pitch", "r_shoulder_roll", "r_shoulder_yaw", "r_elbow",
    "l_shoulder_pitch", "l_shoulder_roll", "l_shoulder_yaw", "l_elbow",
    "l_hip_pitch", "l_hip_roll", "l_hip_yaw", "l_knee", "l_ankle_pitch", "l_ankle_roll",
    "r_hip_pitch", "r_hip_roll", "r_hip_yaw", "r_knee", "r_ankle_pitch", "r_ankle_roll",
]
N_JOINTS = len(URDF_JOINT_ORDER)   # 26

# From retargeted walking data: root z in [0.540, 0.597] m
TARGET_HEIGHT  = 0.57     # m — nominal standing height
MIN_HEIGHT     = 0.35     # m — below this = fallen
MAX_HEIGHT     = 0.80     # m — above this = numerical explosion

# PD gains for position actuators
KP = 200.0   # Nm/rad — position gain
KV = 20.0    # Nm·s/rad — velocity gain (biasprm[2])

MAX_EPISODE_STEPS = 1000


def _build_model():
    """Load URDF, inject freejoint + floor + position actuators, compile."""
    spec = mujoco.MjSpec.from_file(str(URDF_PATH))
    spec.meshdir = str(MESH_DIR)

    # Floating base — URDF root is fixed by default
    fj = spec.worldbody.first_body().add_joint()
    fj.type  = mujoco.mjtJoint.mjJNT_FREE
    fj.name  = "root_freejoint"

    # Floor
    floor = spec.worldbody.add_geom()
    floor.type      = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size      = [20.0, 20.0, 0.01]
    floor.pos       = [0.0, 0.0, 0.0]
    floor.name      = "floor"

    # Position actuators — one per joint
    # gainprm[0] = kp, biasprm[1] = -kp (position spring), biasprm[2] = -kv (velocity damper)
    for jname in URDF_JOINT_ORDER:
        act = spec.add_actuator()
        act.name      = f"{jname}_act"
        act.trntype   = mujoco.mjtTrn.mjTRN_JOINT
        act.target    = jname
        act.gaintype  = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = KP
        act.biastype  = mujoco.mjtBias.mjBIAS_AFFINE
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

        # Joint limit arrays (shape: N_JOINTS) for action rescaling
        # Pull from model's jnt_range (skip freejoint at index 0)
        self.joint_lo = self.model.jnt_range[1:, 0].copy()   # (26,)
        self.joint_hi = self.model.jnt_range[1:, 1].copy()   # (26,)
        # Clamp any unlimited joints to ±π
        self.joint_lo = np.clip(self.joint_lo, -math.pi, 0.0)
        self.joint_hi = np.clip(self.joint_hi,  0.0, math.pi)

        obs_dim = N_JOINTS + N_JOINTS + 4 + 3 + 3   # 62
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32
        )

        self._step_count = 0

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_obs(self):
        # qpos layout: [x,y,z, qw,qx,qy,qz, joint_0..joint_25]
        joint_pos  = self.data.qpos[7:].astype(np.float32)       # (26,)
        joint_vel  = self.data.qvel[6:].astype(np.float32)       # (26,)
        base_quat  = self.data.qpos[3:7].astype(np.float32)      # wxyz (4,)
        base_linvel = self.data.qvel[0:3].astype(np.float32)     # (3,)
        base_angvel = self.data.qvel[3:6].astype(np.float32)     # (3,)
        return np.concatenate([joint_pos, joint_vel, base_quat, base_linvel, base_angvel])

    def _action_to_ctrl(self, action):
        """Map normalized [-1,1] action to joint angle targets within limits."""
        return self.joint_lo + (action + 1.0) * 0.5 * (self.joint_hi - self.joint_lo)

    def _compute_reward(self, action):
        height      = float(self.data.qpos[2])
        forward_vel = float(self.data.qvel[0])   # x-axis = forward
        quat        = self.data.qpos[3:7]         # wxyz

        # Roll/pitch penalty: quat[1]=qx (roll), quat[2]=qy (pitch) in wxyz
        upright_penalty = -0.5 * float(np.sum(np.square(quat[1:3])))

        forward_reward = forward_vel * 2.0
        height_bonus   = 1.0 - abs(height - TARGET_HEIGHT)
        alive_bonus    = 0.5
        ctrl_cost      = -0.001 * float(np.sum(np.square(action)))

        return forward_reward + height_bonus + upright_penalty + alive_bonus + ctrl_cost

    def _is_terminated(self):
        height = float(self.data.qpos[2])
        return height < MIN_HEIGHT or height > MAX_HEIGHT

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0

        # Set initial standing pose: root at TARGET_HEIGHT, identity quaternion
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = TARGET_HEIGHT
        self.data.qpos[3] = 1.0   # qw
        self.data.qpos[4:7] = 0.0

        # Small perturbation to break symmetry
        rng = np.random.default_rng(seed)
        self.data.qpos[7:] += rng.uniform(-0.02, 0.02, N_JOINTS)
        self.data.qvel[:]  += rng.uniform(-0.01, 0.01, self.model.nv)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = self._action_to_ctrl(action)

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs        = self._get_obs()
        reward     = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated  = self._step_count >= MAX_EPISODE_STEPS

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
