#!/usr/bin/env python3
"""
Visualize a trained PPO policy on iRonCub.

Usage:
    python scripts/infer_ppo.py logs/ppo_ironcub/best_model.zip
    python scripts/infer_ppo.py logs/ppo_ironcub/best_model.zip --episodes 5
"""

import argparse
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mujoco_track.env import IronCubWalkEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to trained .zip model")
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    model = PPO.load(args.model_path)

    env = IronCubWalkEnv(render_mode="human")
    obs, _ = env.reset()
    total_reward = 0.0
    ep = 0

    print(f"Running {args.episodes} episodes. Close viewer to stop.\n")
    try:
        while ep < args.episodes:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            env.render()
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                ep += 1
                print(f"Episode {ep}: total_reward = {total_reward:.2f}")
                total_reward = 0.0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
