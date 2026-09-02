#!/usr/bin/env python3
"""
Eval script: roll out a trained AMP policy, compare checkpoints, optionally save GIFs.

Runs N episodes per checkpoint in headless mode, prints a comparison table.
With --save_gif, renders via Isaac Lab's tiled camera (EGL offscreen; no display needed).

Usage:
    # metrics only (fast, no render)
    conda run -n polaris python scripts/eval_amp.py \
        --checkpoints logs/amp_ironcub/run2/checkpoints/best_agent.pt \
                      logs/amp_ironcub/run2/checkpoints/agent_430000.pt \
        --num_episodes 5

    # metrics + GIF (slower — spins up renderer)
    conda run -n polaris python scripts/eval_amp.py \
        --checkpoints logs/amp_ironcub/run2/checkpoints/best_agent.pt \
                      logs/amp_ironcub/run2/checkpoints/agent_430000.pt \
        --save_gif --gif_out eval_compare.gif --num_episodes 3
"""

import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", required=True,
                    help="One or more checkpoint .pt files to compare")
parser.add_argument("--num_episodes", type=int, default=5,
                    help="Episodes to roll out per checkpoint")
parser.add_argument("--num_envs",    type=int, default=4,
                    help="Parallel envs (keep small for eval; 4 gives 4 ep/rollout)")
parser.add_argument("--save_gif",    action="store_true",
                    help="Save a GIF of the rollout (one per checkpoint)")
parser.add_argument("--gif_fps",     type=int, default=30)
parser.add_argument("--gif_out",     type=str, default="eval.gif",
                    help="Output GIF filename (checkpoint name appended if multiple)")
parser.add_argument("--max_steps",   type=int, default=300,
                    help="Max steps per episode (matches training episode length)")
args, _ = parser.parse_known_args()

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=args.save_gif)
simulation_app = app_launcher.app

# ── All other imports after AppLauncher ──────────────────────────────────────
import torch
import numpy as np
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
from skrl.resources.preprocessors.torch import RunningStandardScaler


# ── Camera support (only imported when --save_gif) ───────────────────────────
def _make_camera_cfg():
    """Returns a TiledCameraCfg positioned to view the robot."""
    try:
        from isaaclab.sensors import TiledCameraCfg
        import isaaclab.sim as sim_utils
        return TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/EvalCamera",
            offset=TiledCameraCfg.OffsetCfg(
                pos=(3.5, 0.0, 1.8),
                rot=(0.9135, 0.0, 0.4067, 0.0),  # ~45° tilt looking at robot
                convention="world",
            ),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1_000.0),
            ),
            width=640,
            height=480,
        )
    except Exception as e:
        print(f"  [warn] Camera setup failed: {e} — GIF will not be saved.")
        return None


# ── Model stubs (same arch as training) ─────────────────────────────────────

class Policy(GaussianMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True,
                               min_log_std=-20.0, max_log_std=2.0)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 256),                   nn.ELU(),
            nn.Linear(256, self.num_actions),
        )
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 512), nn.ELU(),
            nn.Linear(512, 256),                   nn.ELU(),
            nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


class Discriminator(DeterministicMixin, Model):
    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        import torch.nn as nn
        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256), nn.ELU(),
            nn.Linear(256, 128),                   nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {}


# ── Rollout one checkpoint ────────────────────────────────────────────────────

def eval_checkpoint(ckpt_path: str, env, wrapped_env, device: str,
                    num_episodes: int, max_steps: int,
                    save_gif: bool = False, gif_path: str = None):
    policy_obs_space = gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(OBS_DIM,))
    amp_obs_space    = gymnasium.spaces.Box(-float("inf"), float("inf"), shape=(AMP_DIM,))
    act_space        = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACT_DIM,))

    models = {
        "policy":        Policy(policy_obs_space, act_space, device),
        "value":         Value(policy_obs_space, act_space, device),
        "discriminator": Discriminator(amp_obs_space, act_space, device),
    }
    rollout_memory = RandomMemory(memory_size=16, num_envs=wrapped_env.num_envs, device=device)
    motion_dataset = RandomMemory(memory_size=1_000, num_envs=1, device=device)
    reply_buffer   = RandomMemory(memory_size=1_000, num_envs=1, device=device)

    cfg = AMP_CFG(
        rollouts=16, learning_epochs=1, mini_batches=1,
        discount_factor=0.99, gae_lambda=0.95, learning_rate=1e-4,
        amp_batch_size=64, discriminator_batch_size=-1,
        discriminator_gradient_penalty_scale=20.0,
        discriminator_logit_regularization_scale=0.05,
        discriminator_weight_decay_scale=1e-4,
        discriminator_loss_scale=1.0,
        task_reward_scale=0.5, style_reward_scale=0.5,
        observation_preprocessor=RunningStandardScaler,
        observation_preprocessor_kwargs={"size": policy_obs_space, "device": device},
        value_preprocessor=RunningStandardScaler,
        value_preprocessor_kwargs={"size": 1, "device": device},
        amp_observation_preprocessor=RunningStandardScaler,
        amp_observation_preprocessor_kwargs={"size": amp_obs_space, "device": device},
        experiment=ExperimentCfg(
            directory="logs/eval_tmp", experiment_name="eval",
            write_interval=99999, checkpoint_interval=99999,
        ),
    )

    agent = AMP(
        models=models, memory=rollout_memory, cfg=cfg,
        observation_space=policy_obs_space, action_space=act_space,
        amp_observation_space=amp_obs_space,
        motion_dataset=motion_dataset, reply_buffer=reply_buffer,
        collect_reference_motions=env.collect_reference_motions,
        device=device,
    )
    agent.load(ckpt_path)
    # Put all models in PyTorch eval mode (no dropout/batchnorm updates)
    for m in models.values():
        m.eval()

    # Grab the observation preprocessor (RunningStandardScaler trained alongside the policy).
    # SKRL 2.x stores it as _observation_preprocessor (private, underscore prefix).
    # Bypassing it feeds raw-scale obs to a policy that expects normalised inputs → robot falls immediately.
    obs_pre = agent._observation_preprocessor  # RunningStandardScaler with fitted mean/std

    ep_lengths = []
    ep_rewards = []
    frames = []

    # SKRL 2.x IsaacLabWrapper.reset() returns a flat tensor (not a dict).
    # IsaacLabWrapper.step() also returns flat tensors; reward/terminated/truncated are (N,1).
    obs, _ = wrapped_env.reset()

    ep_len  = torch.zeros(wrapped_env.num_envs, device=device)
    ep_rew  = torch.zeros(wrapped_env.num_envs, device=device)
    completed = 0

    with torch.no_grad():
        step = 0
        while completed < num_episodes and step < num_episodes * max_steps:
            # Apply the trained RunningStandardScaler (train=False → inference only, no stat update).
            # Without this the policy gets raw-scale observations and outputs garbage actions.
            try:
                obs_scaled = obs_pre(obs)   # train=False is the default
            except Exception:
                obs_scaled = obs

            # policy act — SKRL 2.x returns (actions, outputs_dict)
            result = agent.policy.act({"observations": obs_scaled}, role="policy")
            actions = result[0].clamp(-1.0, 1.0)

            obs, reward, terminated, truncated, info = wrapped_env.step(actions)
            # terminated/truncated are (N,1) tensors; squeeze to (N,) for indexing
            done = (terminated | truncated).squeeze(-1)

            ep_len += 1
            ep_rew += reward.squeeze(-1)  # reward is (N,1) from SKRL wrapper → squeeze to (N,)

            # Collect camera frame (only env 0)
            if save_gif and step % 2 == 0:  # every other step to keep GIF size down
                try:
                    cam = env.scene["eval_camera"]
                    cam.update(dt=0.0)
                    rgb = cam.data.output["rgb"]
                    # rgb: (num_envs, H, W, 4) RGBA; take env 0, drop alpha
                    frame = rgb[0, :, :, :3].cpu().numpy().astype(np.uint8)
                    frames.append(frame)
                except Exception as cam_err:
                    if step == 0:
                        print(f"  [warn] Camera capture failed: {cam_err} — no GIF frames")
                    pass

            # Check done envs
            done_idx = done.nonzero(as_tuple=True)[0]
            for idx in done_idx:
                if completed < num_episodes:
                    ep_lengths.append(ep_len[idx].item())
                    ep_rewards.append(ep_rew[idx].item())
                    completed += 1
                ep_len[idx] = 0
                ep_rew[idx] = 0.0

            step += 1

    # Save GIF
    if save_gif and frames and gif_path:
        try:
            import imageio
            imageio.mimsave(gif_path, frames, fps=args.gif_fps, loop=0)
            print(f"  GIF saved → {gif_path}  ({len(frames)} frames)")
        except ImportError:
            print("  [warn] imageio not found; install with: pip install imageio")
        except Exception as e:
            print(f"  [warn] GIF save failed: {e}")

    return {
        "ep_length_mean": np.mean(ep_lengths) if ep_lengths else 0.0,
        "ep_length_max":  np.max(ep_lengths)  if ep_lengths else 0.0,
        "ep_length_min":  np.min(ep_lengths)  if ep_lengths else 0.0,
        "ep_reward_mean": np.mean(ep_rewards) if ep_rewards else 0.0,
        "n_episodes":     completed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build env once, reuse for all checkpoints
    env_cfg = IronCubAMPEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.robot_cfg = IRONCUB_CFG

    # Optionally add camera
    if args.save_gif:
        cam_cfg = _make_camera_cfg()
        if cam_cfg is not None:
            env_cfg.scene.eval_camera = cam_cfg
        else:
            args.save_gif = False  # fallback

    raw_env = IronCubAMPEnv(cfg=env_cfg, render_mode=None)
    wrapped = wrap_env(raw_env, wrapper="isaaclab")

    results = {}
    for ckpt in args.checkpoints:
        ckpt = str(ckpt)
        name = Path(ckpt).stem
        print(f"\n─── Evaluating: {name} ───")

        gif_path = None
        if args.save_gif:
            base = Path(args.gif_out)
            gif_path = str(base.parent / f"{base.stem}_{name}{base.suffix}")

        res = eval_checkpoint(
            ckpt_path=ckpt, env=raw_env, wrapped_env=wrapped,
            device=device, num_episodes=args.num_episodes,
            max_steps=args.max_steps, save_gif=args.save_gif,
            gif_path=gif_path,
        )
        results[name] = res

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"{'Checkpoint':<22} {'ep_len mean':>11} {'ep_len max':>10} {'ep_rew mean':>11} {'n_ep':>5}")
    print("─" * 62)
    for name, r in results.items():
        print(f"  {name:<20} {r['ep_length_mean']:>11.1f} {r['ep_length_max']:>10.0f} "
              f"{r['ep_reward_mean']:>11.1f} {r['n_episodes']:>5}")
    print("═" * 62)

    if len(results) == 2:
        names = list(results.keys())
        delta = results[names[1]]["ep_length_mean"] - results[names[0]]["ep_length_mean"]
        winner = names[1] if delta > 0 else names[0]
        print(f"\nBetter checkpoint (ep_length): {winner}  (Δ {abs(delta):.1f} steps)")

    simulation_app.close()


if __name__ == "__main__":
    main()
