# Eval results — run2

Two checkpoints evaluated after 430K training steps.

| Checkpoint | ep_len mean | ep_len max | ep_rew mean |
|---|---|---|---|
| best_agent (~50K steps) | 241.8 | 299 | 1239.4 |
| agent_430000 (430K steps) | 240.8 | 299 | 1217.8 |

- 5 episodes × 4 envs each, 299-step episode cap
- Robot starts from Reference State Initialization (random motion-clip frame)
- Discriminator collapsed at run2 (disc_loss ≈ 0.021): policy survives but style signal is dead
- GIFs: ~12.5 fps, 320×240

## Hyperparams (run2)
- `discriminator_gradient_penalty_scale = 10.0`
- `discriminator_loss_scale = 2.5`
- `style_w = 0.6`, `task_w = 0.4`
- Resumed from run1 checkpoint at 10K steps

## Next steps
- run3: GP=20, disc_loss_scale=1.0, discriminator weights reset from scratch
