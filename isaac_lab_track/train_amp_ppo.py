#!/usr/bin/env python3
"""
AMP + PPO training for iRonCub walking — SKRL 2.x.

The AMP discriminator is trained to distinguish robot states from the
retargeted walking motion prior. PPO handles the policy update.
Style reward (discriminator) and task reward (alive + forward) are
combined via task_reward_scale / style_reward_scale.

Usage:
    # from repo root:
    OMNI_KIT_ACCEPT_EULA=Y conda run -n polaris --no-capture-output \\
        python isaac_lab_track/train_amp_ppo.py --headless --num_envs 64
    python isaac_lab_track/train_amp_ppo.py --headless --num_envs 4096
    python isaac_lab_track/train_amp_ppo.py --checkpoint logs/amp_ironcub/agent_50000.pt
"""

import argparse
import sys
from pathlib import Path

# ── AppLauncher must be created before any omni / Isaac Lab imports ───────────
parser = argparse.ArgumentParser()
parser.add_argument("--headless",   action="store_true")
parser.add_argument("--num_envs",   type=int, default=64)
parser.add_argument("--timesteps",  type=int, default=50_000_000)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Resume from a SKRL agent checkpoint (.pt)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

# ── All other imports after AppLauncher ───────────────────────────────────────
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from isaac_lab_track.ironcub_cfg import IRONCUB_CFG
from isaac_lab_track.amp_env import IronCubAMPEnv, IronCubAMPEnvCfg, OBS_DIM, AMP_DIM, ACT_DIM

from skrl.agents.torch.amp import AMP, AMP_CFG
from skrl.agents.torch.base import ExperimentCfg
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler

import gymnasium

LOG_DIR = Path("logs/amp_ironcub")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────

class Policy(GaussianMixin, Model):
    """Actor — maps policy observation to action distribution (Gaussian)."""
    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0):
        # SKRL 2.x: Model.__init__ is keyword-only
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # SKRL 2.x: inputs key is "states"; return (actions, {"log_std": ...})
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    """Critic — estimates value of policy observation."""
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Discriminator(DeterministicMixin, Model):
    """AMP discriminator — classifies env states vs reference motion states."""
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining on: {device}")

    # ── Environment ───────────────────────────────────────────────────────────
    env_cfg = IronCubAMPEnvCfg()
    # In Isaac Lab v2 num_envs lives in scene, not on the top-level cfg
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.robot_cfg = IRONCUB_CFG

    raw_env = IronCubAMPEnv(cfg=env_cfg, render_mode=None)
    env     = wrap_env(raw_env, wrapper="isaaclab")

    # ── Observation / action spaces for models ────────────────────────────────
    policy_obs_space = gymnasium.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(OBS_DIM,))
    amp_obs_space    = gymnasium.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(AMP_DIM,))
    act_space        = gymnasium.spaces.Box(
        low=-1.0, high=1.0, shape=(ACT_DIM,))

    # ── Models ────────────────────────────────────────────────────────────────
    models = {
        "policy":        Policy(policy_obs_space, act_space, device),
        "value":         Value(policy_obs_space, act_space, device),
        "discriminator": Discriminator(amp_obs_space, act_space, device),
    }

    # ── Memories ──────────────────────────────────────────────────────────────
    rollout_memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)

    # Motion dataset: filled at init with reference frames from the motion prior.
    # RandomMemory with num_envs=1 accepts bulk add_samples((N, dim)) tensors.
    MOTION_DATASET_SIZE = 200_000
    motion_dataset = RandomMemory(memory_size=MOTION_DATASET_SIZE, num_envs=1, device=device)

    # Replay buffer: AMP uses this to prevent discriminator overfitting
    REPLAY_BUFFER_SIZE = 1_000_000
    reply_buffer = RandomMemory(memory_size=REPLAY_BUFFER_SIZE, num_envs=1, device=device)

    # ── AMP agent config (SKRL 2.x: dataclass, not dict) ─────────────────────
    cfg = AMP_CFG(
        # PPO hyperparameters
        rollouts=16,
        learning_epochs=6,
        mini_batches=2,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=5e-5,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_loss_scale=2.0,
        entropy_loss_scale=0.0,

        # AMP discriminator
        amp_batch_size=512,
        discriminator_batch_size=-1,           # use full amp_batch_size
        discriminator_gradient_penalty_scale=10.0,
        discriminator_logit_regularization_scale=0.05,
        discriminator_weight_decay_scale=1e-4,
        discriminator_loss_scale=5.0,

        # Reward mixing
        task_reward_scale=0.3,
        style_reward_scale=0.7,

        # Observation preprocessing
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": policy_obs_space, "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        amp_observation_preprocessor=RunningStandardScaler,
        amp_observation_preprocessor_kwargs={"size": amp_obs_space, "device": device},

        # Logging / saving
        experiment=ExperimentCfg(
            directory=str(LOG_DIR),
            experiment_name="amp_ironcub",
            write_interval=100,
            checkpoint_interval=1000,
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

    # ── Trainer ───────────────────────────────────────────────────────────────
    from skrl.trainers.torch import SequentialTrainerCfg
    trainer_cfg = SequentialTrainerCfg(
        timesteps=args.timesteps,
        headless=args.headless,
    )
    trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

    print(f"Starting AMP+PPO  |  {args.num_envs} envs  |  {args.timesteps:,} steps\n")
    trainer.train()

    simulation_app.close()


if __name__ == "__main__":
    main()
