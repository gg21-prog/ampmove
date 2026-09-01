#!/usr/bin/env python3
"""
AMP + PPO training for iRonCub walking — SKRL 2.x.

Hyperparameters tuned against the MuJoCo baseline that achieved walking:
  - Survival >> style: alive_w=5, upright_w=3, forward gated by upright
  - learning_rate=1e-4 (standard Isaac Lab humanoid scale)
  - task/style = 0.5/0.5 (balanced; shift to 0.3/0.7 once walking emerges)
  - discriminator_gradient_penalty_scale=5.0 (AMP paper default, not 10)
  - timesteps default = 1_000_000 (~10 hrs @ 4096 envs on RTX 4090 / 28 it/s)

Usage:
    OMNI_KIT_ACCEPT_EULA=Y PYTHONUNBUFFERED=1 conda run -n polaris --no-capture-output \\
        python isaac_lab_track/train_amp_ppo.py --headless --num_envs 4096 --wandb

    # quick smoke test
    python isaac_lab_track/train_amp_ppo.py --headless --num_envs 4 --timesteps 200

    # resume
    python isaac_lab_track/train_amp_ppo.py --headless --checkpoint logs/amp_ironcub/...

Projected wall time (RTX 4090, 4096 envs, ~28 it/s):
    73K steps  →  43 min  →  ~300M env transitions (AMP paper baseline)
    200K steps →   2 hrs  →  ~820M env transitions (motion style emerging)
    500K steps →   5 hrs  →    ~2B env transitions  (clear walking)
    1M steps   →  10 hrs  →    ~4B env transitions  (good convergence)
"""

import argparse
import sys
from pathlib import Path

# ── AppLauncher must be created before any omni / Isaac Lab imports ───────────
parser = argparse.ArgumentParser()
parser.add_argument("--headless",    action="store_true")
parser.add_argument("--num_envs",    type=int,   default=4096)
parser.add_argument("--timesteps",   type=int,   default=1_000_000,
                    help="Trainer steps (~28 it/s × 4096 envs on RTX 4090)")
parser.add_argument("--checkpoint",  type=str,   default=None,
                    help="Resume from a SKRL checkpoint (.pt)")
parser.add_argument("--wandb",       action="store_true",
                    help="Enable Weights & Biases logging (requires: pip install wandb)")
parser.add_argument("--run_name",    type=str,   default="amp_ironcub",
                    help="W&B / experiment name")
parser.add_argument("--lr",          type=float, default=1e-4,
                    help="Policy + value learning rate (default 1e-4)")
parser.add_argument("--task_w",      type=float, default=0.5,
                    help="Task reward weight in AMP total reward (default 0.5)")
parser.add_argument("--style_w",     type=float, default=0.5,
                    help="Style (discriminator) reward weight (default 0.5)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── All other imports after AppLauncher ───────────────────────────────────────
import torch
import torch.nn as nn
import gymnasium

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from isaac_lab_track.ironcub_cfg import IRONCUB_CFG
from isaac_lab_track.amp_env import IronCubAMPEnv, IronCubAMPEnvCfg, OBS_DIM, AMP_DIM, ACT_DIM

from skrl.agents.torch.amp import AMP, AMP_CFG
from skrl.agents.torch.base import ExperimentCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
from skrl.resources.preprocessors.torch import RunningStandardScaler

LOG_DIR = Path("logs/amp_ironcub")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Model definitions ─────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    """Actor — Gaussian policy over joint position targets."""
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-20.0, max_log_std=2.0)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 256),                   nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # SKRL 2.x: key is "observations"; return (actions, {"log_std": ...})
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    """Critic — state value estimator."""
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 256),                   nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Discriminator(DeterministicMixin, Model):
    """AMP discriminator — real (motion prior) vs fake (policy rollout)."""
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128),                   nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = IronCubAMPEnvCfg()
    env_cfg.scene.num_envs = args.num_envs   # num_envs lives in scene in Isaac Lab v2
    env_cfg.robot_cfg = IRONCUB_CFG
    raw_env = IronCubAMPEnv(cfg=env_cfg, render_mode=None)
    env     = wrap_env(raw_env, wrapper="isaaclab")

    # ── Observation / action spaces ───────────────────────────────────────────
    policy_obs_space = gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(OBS_DIM,))
    amp_obs_space    = gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(AMP_DIM,))
    act_space        = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,))

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "policy":        Policy(policy_obs_space, act_space, device),
        "value":         Value(policy_obs_space, act_space, device),
        "discriminator": Discriminator(amp_obs_space, act_space, device),
    }

    # ── Memories ──────────────────────────────────────────────────────────────
    rollout_memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

    # motion_dataset: filled at init; num_envs=1 accepts bulk (N, dim) add_samples
    motion_dataset = RandomMemory(memory_size=200_000, num_envs=1, device=device)
    reply_buffer   = RandomMemory(memory_size=1_000_000, num_envs=1, device=device)

    # ── AMP config (SKRL 2.x dataclass) ──────────────────────────────────────
    cfg = AMP_CFG(
        # PPO
        rollouts=16,
        learning_epochs=6,
        mini_batches=2,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=args.lr,          # default 1e-4 (was 5e-5 — too slow)
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_loss_scale=2.0,
        entropy_loss_scale=0.0,

        # AMP discriminator — using AMP paper defaults
        amp_batch_size=512,
        discriminator_batch_size=-1,
        discriminator_gradient_penalty_scale=5.0,   # AMP paper default (was 10 — too aggressive)
        discriminator_logit_regularization_scale=0.05,
        discriminator_weight_decay_scale=1e-4,
        discriminator_loss_scale=5.0,

        # Reward mixing — balanced to start; CLI args let you tune without editing
        task_reward_scale=args.task_w,   # default 0.5
        style_reward_scale=args.style_w, # default 0.5

        # Observation preprocessing
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": policy_obs_space, "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        amp_observation_preprocessor=RunningStandardScaler,
        amp_observation_preprocessor_kwargs={"size": amp_obs_space, "device": device},

        # Logging
        experiment=ExperimentCfg(
            directory=str(LOG_DIR),
            experiment_name=args.run_name,
            write_interval=200,       # every 200 steps (~7s at 28 it/s)
            checkpoint_interval=5000, # every 5K steps (~3 min)
            wandb=args.wandb,
            wandb_kwargs={"project": "ampmove-ironcub", "entity": "gg21", "name": args.run_name} if args.wandb else {},
        ),
    )

    agent = AMP(
        models=models,
        memory=rollout_memory,
        cfg=cfg,
        observation_space=policy_obs_space,
        action_space=act_space,
        amp_observation_space=amp_obs_space,
        motion_dataset=motion_dataset,
        reply_buffer=reply_buffer,
        collect_reference_motions=raw_env.collect_reference_motions,
        device=device,
    )

    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        agent.load(args.checkpoint)

    # ── Projected time estimate ───────────────────────────────────────────────
    ITS = 28  # measured it/s at 4096 envs on RTX 4090
    secs = args.timesteps / ITS
    hrs  = secs / 3600
    env_transitions = args.timesteps * args.num_envs
    print(f"\nTraining on: {device}")
    print(f"  {args.num_envs} envs  |  {args.timesteps:,} trainer steps")
    print(f"  ~{env_transitions/1e9:.1f}B env transitions")
    if args.num_envs == 4096:
        print(f"  Estimated wall time: {hrs:.1f} hrs  ({secs/60:.0f} min) at ~{ITS} it/s")
    print(f"  task_w={args.task_w}  style_w={args.style_w}  lr={args.lr}")
    print(f"  W&B: {'enabled → gg21/ampmove-ironcub/' + args.run_name if args.wandb else 'disabled (pass --wandb to enable)'}\n")

    trainer = SequentialTrainer(
        cfg=SequentialTrainerCfg(timesteps=args.timesteps, headless=args.headless),
        env=env,
        agents=agent,
    )
    trainer.train()
    simulation_app.close()


if __name__ == "__main__":
    main()
