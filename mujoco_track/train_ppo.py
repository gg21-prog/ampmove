#!/usr/bin/env python3
"""
PPO baseline for iRonCub walking — Stable Baselines3.

Usage:
    python scripts/train_ppo.py              # train from scratch
    python scripts/train_ppo.py --timesteps 3000000
    python scripts/train_ppo.py --resume logs/ppo_ironcub/best_model.zip
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mujoco_track.env import IronCubWalkEnv

LOG_DIR  = Path("logs/ppo_ironcub")
SAVE_DIR = Path("checkpoints/ppo_ironcub")


class TerminalLogger(BaseCallback):
    """Prints episode stats every N steps — matches quadmove style."""
    def __init__(self, log_freq=2000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._ep_rewards = []
        self._ep_lengths = []
        self._cur_rew = 0.0
        self._cur_len = 0

    def _on_step(self):
        self._cur_rew += float(self.locals["rewards"][0])
        self._cur_len += 1
        if self.locals["dones"][0]:
            self._ep_rewards.append(self._cur_rew)
            self._ep_lengths.append(self._cur_len)
            self._cur_rew = 0.0
            self._cur_len = 0
        if self.n_calls % self.log_freq == 0 and self._ep_rewards:
            mean_r = np.mean(self._ep_rewards[-10:])
            mean_l = np.mean(self._ep_lengths[-10:])
            print(
                f"Step {self.num_timesteps:>8d} | "
                f"MeanReward(10ep): {mean_r:7.2f} | "
                f"MeanLen: {mean_l:6.1f} | "
                f"Episodes: {len(self._ep_rewards)}"
            )
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .zip checkpoint to resume from")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="Number of parallel envs (default 4)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Sanity check single env before vectorizing
    print("Checking environment...")
    check_env(IronCubWalkEnv(), warn=True)
    print("Environment OK.\n")

    # Vectorized training envs
    train_env = make_vec_env(IronCubWalkEnv, n_envs=args.n_envs)

    # Separate eval env (single, no Monitor wrapping needed — EvalCallback adds it)
    eval_env = Monitor(IronCubWalkEnv())

    callbacks = [
        TerminalLogger(log_freq=2000),
        EvalCallback(
            eval_env,
            best_model_save_path=str(LOG_DIR),
            log_path=str(LOG_DIR),
            eval_freq=max(10_000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=str(SAVE_DIR),
            name_prefix="ppo_ironcub",
            verbose=1,
        ),
    ]

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=train_env)
    else:
        model = PPO(
            policy          = "MlpPolicy",
            env             = train_env,
            device          = "cpu",
            learning_rate   = 3e-4,
            n_steps         = 2048,
            batch_size      = 64,
            n_epochs        = 10,
            gamma           = 0.99,
            gae_lambda      = 0.95,
            clip_range      = 0.2,
            ent_coef        = 0.0,
            vf_coef         = 0.5,
            max_grad_norm   = 0.5,
            tensorboard_log = str(LOG_DIR / "tb"),
            verbose         = 1,
        )

    print(f"Training for {args.timesteps:,} timesteps across {args.n_envs} envs...\n")
    model.learn(
        total_timesteps  = args.timesteps,
        callback         = callbacks,
        reset_num_timesteps = args.resume is None,
        progress_bar     = True,
    )

    model.save(str(LOG_DIR / "final_model"))
    print(f"\nSaved final model → {LOG_DIR / 'final_model.zip'}")


if __name__ == "__main__":
    main()
