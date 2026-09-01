#!/usr/bin/env python3
"""
AMP + PPO training for iRonCub walking — SKRL.

The AMP discriminator is trained to distinguish robot states from the
retargeted walking motion prior. PPO handles the policy update.
Style reward (discriminator) and task reward (alive + forward) are
combined with configurable weights.

Usage:
    # from repo root:
    python isaac_lab_track/train_amp_ppo.py
    python isaac_lab_track/train_amp_ppo.py --headless --num_envs 4096
    python isaac_lab_track/train_amp_ppo.py --checkpoint logs/amp_ironcub/agent_50000.pt
"""

import argparse
import sys
from pathlib import Path

# ── AppLauncher must be created before any omni / Isaac Lab imports ───────────
parser = argparse.ArgumentParser()
parser.add_argument("--headless",    action="store_true")
parser.add_argument("--num_envs",    type=int, default=4096)
parser.add_argument("--timesteps",   type=int, default=50_000_000)
parser.add_argument("--checkpoint",  type=str, default=None,
                    help="Resume from a SKRL agent checkpoint (.pt)")
args, _ = parser.parse_known_args()

from omni.isaac.lab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── All other imports after AppLauncher ───────────────────────────────────────
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from isaac_lab_track.ironcub_cfg import IRONCUB_CFG
from isaac_lab_track.amp_env import IronCubAMPEnv, IronCubAMPEnvCfg, OBS_DIM, AMP_DIM, ACT_DIM

from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler

LOG_DIR  = Path("logs/amp_ironcub")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    """Actor — maps policy observation to action distribution (Gaussian)."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["obs"]), self.log_std_parameter, {}


class Value(DeterministicMixin, Model):
    """Critic — estimates value of policy observation."""
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["obs"]), {}


class Discriminator(DeterministicMixin, Model):
    """AMP discriminator — classifies env states vs reference motion states."""
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["obs"]), {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = IronCubAMPEnvCfg()
    env_cfg.num_envs  = args.num_envs
    env_cfg.robot_cfg = IRONCUB_CFG

    raw_env = IronCubAMPEnv(cfg=env_cfg, render_mode=None)
    env     = wrap_env(raw_env, wrapper="isaaclab")

    # ── Observation / action spaces for models ────────────────────────────────
    # Policy obs space (62-dim) and AMP obs space (64-dim) may differ
    import gymnasium
    policy_obs_space = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"),
                                             shape=(OBS_DIM,))
    amp_obs_space    = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"),
                                             shape=(AMP_DIM,))
    act_space        = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,))

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "policy":        Policy(policy_obs_space, act_space, device),
        "value":         Value(policy_obs_space, act_space, device),
        "discriminator": Discriminator(amp_obs_space, act_space, device),
    }

    # ── Memory ────────────────────────────────────────────────────────────────
    rollout_memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

    # ── AMP agent config ──────────────────────────────────────────────────────
    cfg = AMP_DEFAULT_CONFIG.copy()
    cfg.update({
        # PPO hyperparameters
        "rollouts":         16,
        "learning_epochs":  6,
        "mini_batches":     2,
        "discount_factor":  0.99,
        "lambda":           0.95,
        "learning_rate":    5e-5,
        "grad_norm_clip":   1.0,
        "ratio_clip":       0.2,
        "value_loss_scale": 2.0,
        "entropy_loss_scale": 0.0,

        # AMP discriminator
        "amp_batch_size":                          512,
        "amp_replay_buffer_size":                  1_000_000,
        "amp_discriminator_update_epochs":         5,
        "amp_discriminator_learning_rate":         5e-5,
        "amp_discriminator_gradient_penalty_scale": 10.0,

        # Reward mixing (AMP-prefixed keys match SKRL AMP_DEFAULT_CONFIG)
        "amp_task_reward_weight":  0.3,
        "amp_style_reward_weight": 0.7,

        # Observation preprocessing
        "state_preprocessor":       RunningStandardScaler,
        "state_preprocessor_kwargs": {"size": policy_obs_space, "device": device},
        "value_preprocessor":       RunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": device},
        "amp_state_preprocessor":       RunningStandardScaler,
        "amp_state_preprocessor_kwargs": {"size": amp_obs_space, "device": device},

        # Logging / saving
        "experiment": {
            "directory":          str(LOG_DIR),
            "experiment_name":    "amp_ironcub",
            "write_interval":     100,
            "checkpoint_interval": 1000,
        },
    })

    agent = AMP(
        models=models,
        memory=rollout_memory,
        cfg=cfg,
        observation_space=policy_obs_space,
        action_space=act_space,
        amp_observation_space=amp_obs_space,
        device=device,
    )

    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer_cfg = {
        "timesteps": args.timesteps,
        "headless":  args.headless,
    }
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print(f"Starting AMP+PPO training | {args.num_envs} envs | {args.timesteps:,} steps\n")
    trainer.train()

    simulation_app.close()


if __name__ == "__main__":
    main()
