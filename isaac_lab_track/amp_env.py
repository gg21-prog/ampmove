#!/usr/bin/env python3
"""
Isaac Lab DirectRLEnv for iRonCub AMP training.

Policy observation (62-dim):
    dof_pos(26) | dof_vel(26) | base_lin_vel_b(3) | base_ang_vel_b(3)
    | projected_gravity_b(3) | phase(1)

AMP state (64-dim) — discriminator input, must match collect_reference_motions():
    root_lin_vel_w(3) | root_ang_vel_b(3) | dof_pos(26) | dof_vel(26)
    | l_foot_pos_root(3) | r_foot_pos_root(3)

Action (26-dim): normalized joint position targets in [-1, 1].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import gymnasium
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

REPO_ROOT   = Path(__file__).resolve().parent.parent
MOTION_PATH = REPO_ROOT / "motion_priors/walking/07_12_isaaclab.npz"

N_DOF       = 26
OBS_DIM     = 62
AMP_DIM     = 64
ACT_DIM     = N_DOF

PHYSICS_DT  = 1.0 / 120.0
CONTROL_DT  = 1.0 / 30.0
DECIMATION  = round(CONTROL_DT / PHYSICS_DT)   # = 4

MIN_HEIGHT  = 0.35
MAX_HEIGHT  = 0.90

FOOT_NAMES  = ["l_foot", "r_foot"]


@configclass
class IronCubAMPEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        gravity=(0.0, 0.0, -9.81),
    )
    # DirectRLEnv requires scene to set num_envs and env_spacing
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True
    )
    decimation: int = DECIMATION
    episode_length_s: float = 10.0

    observation_space: int = OBS_DIM
    action_space: int = ACT_DIM
    state_space: int = 0

    # robot — caller sets this via IronCubAMPEnvCfg(robot_cfg=IRONCUB_CFG)
    robot_cfg: ArticulationCfg = None   # type: ignore

    # reward weights — tuned to match MuJoCo baseline philosophy:
    #   survival >> style, forward only rewarded when upright
    alive_w:   float = 5.0    # strong survival signal
    forward_w: float = 0.5    # forward bonus, but gated by upright in _get_rewards
    upright_w: float = 3.0    # upright posture (matches MuJoCo exp(-5*tilt) weight)
    ctrl_w:    float = 0.001  # small action regularisation


class IronCubAMPEnv(DirectRLEnv):
    cfg: IronCubAMPEnvCfg

    def __init__(self, cfg: IronCubAMPEnvCfg, render_mode: str | None = None, **kwargs):
        # Load npz before super().__init__ so data is ready for _setup_scene
        self._load_motion_prior()

        super().__init__(cfg, render_mode, **kwargs)

        # After super().__init__() the scene is set up and sim has been reset —
        # robot.joint_names is now available.
        self._build_joint_mapping()

        # Gait phase counter per env
        self._phase = torch.zeros(self.num_envs, 1, device=self.device)
        # Phase advances by this fraction of cycle per control step
        self._phase_step = CONTROL_DT / (self._n_frames / self._motion_fps)

        # Joint limit cache (populated on first action)
        self._joint_lo: torch.Tensor | None = None
        self._joint_hi: torch.Tensor | None = None

        # Zero-init so _get_rewards doesn't crash before first _pre_physics_step
        self._actions      = torch.zeros(self.num_envs, ACT_DIM, device=self.device)
        self._joint_targets = torch.zeros(self.num_envs, N_DOF,   device=self.device)

        self._amp_obs_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(AMP_DIM,), dtype=np.float32
        )

    # ── Motion prior loading ───────────────────────────────────────────────────

    def _load_motion_prior(self):
        d = np.load(MOTION_PATH)
        self._npz_root_vel     = torch.tensor(d["root_vel"],     dtype=torch.float32)  # (N,3)
        self._npz_root_ang_vel = torch.tensor(d["root_ang_vel"], dtype=torch.float32)  # (N,3)
        self._npz_dof_pos      = torch.tensor(d["dof_pos"],      dtype=torch.float32)  # (N,26)
        self._npz_dof_vel      = torch.tensor(d["dof_vel"],      dtype=torch.float32)  # (N,26)
        self._npz_root_pos     = torch.tensor(d["root_pos"],     dtype=torch.float32)  # (N,3)
        self._npz_root_rot     = torch.tensor(d["root_rot"],     dtype=torch.float32)  # (N,4) wxyz
        self._npz_key_body_pos = torch.tensor(d["key_body_pos"], dtype=torch.float32)  # (N,2,3)
        self._motion_fps       = float(d["fps"])
        self._motion_joint_names = list(d["joint_names"])
        self._n_frames         = self._npz_dof_pos.shape[0]

    # ── Joint mapping (called after super().__init__ when robot is live) ──────

    def _build_joint_mapping(self):
        dof_names    = list(self.robot.joint_names)
        motion_names = self._motion_joint_names

        # For each Isaac Lab DOF j, which index in the motion prior?
        try:
            self._ref2dof = torch.tensor(
                [motion_names.index(n) for n in dof_names], dtype=torch.long
            )
        except ValueError as e:
            raise RuntimeError(
                f"Joint name mismatch: {e}\n"
                f"  Motion joints: {motion_names}\n"
                f"  Asset joints:  {dof_names}"
            ) from e

        # Foot body indices for AMP state computation
        foot_ids, foot_mask = self.robot.find_bodies(FOOT_NAMES)
        if not all(foot_mask):
            raise RuntimeError(f"Could not find foot bodies {FOOT_NAMES} in USD asset.")
        self._foot_ids = torch.tensor(foot_ids, dtype=torch.long)

        # Precompute ALL reference AMP states in Isaac Lab DOF order (done once)
        dof_pos   = self._npz_dof_pos[:, self._ref2dof]   # (N,26)
        dof_vel   = self._npz_dof_vel[:, self._ref2dof]   # (N,26)
        foot_flat = self._npz_key_body_pos.reshape(self._n_frames, -1)  # (N,6)

        self._ref_amp_states = torch.cat([
            self._npz_root_vel,      # (N,3)
            self._npz_root_ang_vel,  # (N,3)
            dof_pos,                  # (N,26)
            dof_vel,                  # (N,26)
            foot_flat,               # (N,6)
        ], dim=-1).to(self.device)   # → (N, 64)

        # Store DOF-ordered motion data for RSI resets
        self._ref_dof_pos      = dof_pos.to(self.device)
        self._ref_dof_vel      = dof_vel.to(self.device)
        self._ref_root_pos     = self._npz_root_pos.to(self.device)
        self._ref_root_rot     = self._npz_root_rot.to(self.device)
        self._ref_root_lin_vel = self._npz_root_vel.to(self.device)

    # ── Scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        # Ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/GroundPlane", ground_cfg)

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)

        # Robot — prim path uses Isaac Lab's per-env wildcard pattern
        robot_cfg = self.cfg.robot_cfg.replace(
            prim_path="/World/envs/env_.*/Robot"
        )
        self.robot = Articulation(robot_cfg)
        self.scene.articulations["robot"] = self.robot

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/GroundPlane"])

    # ── Action processing ─────────────────────────────────────────────────────

    def _get_joint_limits(self):
        if self._joint_lo is None:
            lims = self.robot.data.joint_limits  # (E, 26, 2)
            self._joint_lo = lims[0, :, 0].clamp(-math.pi, 0.0)
            self._joint_hi = lims[0, :, 1].clamp(0.0,      math.pi)
        return self._joint_lo, self._joint_hi

    def _actions_to_targets(self, actions: torch.Tensor) -> torch.Tensor:
        lo, hi = self._get_joint_limits()
        return lo + (actions.clamp(-1.0, 1.0) + 1.0) * 0.5 * (hi - lo)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions       = actions.clone()
        self._joint_targets = self._actions_to_targets(actions)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    # ── Observations ──────────────────────────────────────────────────────────

    def _compute_policy_obs(self) -> torch.Tensor:
        dof_pos   = self.robot.data.joint_pos            # (E,26)
        dof_vel   = self.robot.data.joint_vel            # (E,26)
        lin_vel_b = self.robot.data.root_lin_vel_b       # (E,3)
        ang_vel_b = self.robot.data.root_ang_vel_b       # (E,3)
        grav_b    = self.robot.data.projected_gravity_b  # (E,3)
        phase     = self._phase                          # (E,1)
        return torch.cat([dof_pos, dof_vel, lin_vel_b, ang_vel_b, grav_b, phase], dim=-1)

    def _compute_amp_state(self) -> torch.Tensor:
        dof_pos   = self.robot.data.joint_pos              # (E,26)
        dof_vel   = self.robot.data.joint_vel              # (E,26)
        lin_vel_w = self.robot.data.root_lin_vel_w         # (E,3)
        ang_vel_b = self.robot.data.root_ang_vel_b         # (E,3)

        foot_pos_w = self.robot.data.body_pos_w[:, self._foot_ids, :]  # (E,2,3)
        root_pos_w = self.robot.data.root_pos_w.unsqueeze(1)           # (E,1,3)
        foot_rel   = (foot_pos_w - root_pos_w).reshape(self.num_envs, -1)  # (E,6)

        return torch.cat([lin_vel_w, ang_vel_b, dof_pos, dof_vel, foot_rel], dim=-1)

    def _get_observations(self) -> dict:
        self._phase = (self._phase + self._phase_step) % 1.0
        amp_state = self._compute_amp_state()  # (E,64)
        # SKRL AMP agent reads amp_obs from infos (self.extras), not from obs dict
        self.extras["amp_obs"] = amp_state
        return {
            "policy": self._compute_policy_obs(),  # (E,62)
            "amp":    amp_state,
        }

    # ── Rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        h         = self.robot.data.root_pos_w[:, 2]
        forward_v = self.robot.data.root_lin_vel_w[:, 0]
        grav_b    = self.robot.data.projected_gravity_b

        # Upright score: 1 when vertical, decays with tilt (matches MuJoCo upright gate)
        upright   = torch.exp(-5.0 * (grav_b[:, 0] ** 2 + grav_b[:, 1] ** 2))

        alive     = ((h > MIN_HEIGHT) & (h < MAX_HEIGHT)).float() * self.cfg.alive_w
        # Gate forward by upright: no forward bonus for a falling robot (MuJoCo does this)
        forward   = forward_v.clamp(min=0.0) * upright * self.cfg.forward_w
        upright_r = upright * self.cfg.upright_w
        ctrl_cost = -self.cfg.ctrl_w * self._actions.square().sum(dim=-1)

        return alive + forward + upright_r + ctrl_cost

    # ── Dones ─────────────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        h        = self.robot.data.root_pos_w[:, 2]
        fallen   = (h < MIN_HEIGHT) | (h > MAX_HEIGHT)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, time_out

    # ── Extras ────────────────────────────────────────────────────────────────
    # DirectRLEnv v2 uses self.extras dict rather than _get_extras(); nothing to do.

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return

        # Reset episode length counter (DirectRLEnv has no base impl to call)
        self.episode_length_buf[env_ids] = 0

        n      = len(env_ids)
        frames = torch.randint(0, self._n_frames, (n,), device=self.device)

        # Root state: env origin XY, reference height Z, reference quaternion
        root_state          = self.robot.data.default_root_state[env_ids].clone()
        origins             = self.scene.env_origins[env_ids]           # (n,3)
        root_state[:, 0:3]  = origins
        root_state[:, 2]    = self._ref_root_pos[frames, 2]             # ref height
        root_state[:, 3:7]  = self._ref_root_rot[frames]                # wxyz
        root_state[:, 7:10] = self._ref_root_lin_vel[frames]
        root_state[:, 10:13] = 0.0

        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

        dof_pos = self._ref_dof_pos[frames] + torch.randn(n, N_DOF, device=self.device) * 0.01
        dof_vel = self._ref_dof_vel[frames]
        self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        # Phase from sampled frame position
        self._phase[env_ids] = (frames.float() / self._n_frames).unsqueeze(-1)

    # ── AMP interface for SKRL ─────────────────────────────────────────────────

    @property
    def amp_observation_space(self) -> gymnasium.spaces.Box:
        return self._amp_obs_space

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """Random AMP states from the motion prior — called by SKRL discriminator."""
        idx = torch.randint(0, self._n_frames, (num_samples,), device=self.device)
        return self._ref_amp_states[idx]
