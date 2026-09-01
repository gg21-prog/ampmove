# ampmove

Bipedal locomotion for **iRonCub-Mk1_1** (jet-powered flying humanoid, gbionics), via two parallel RL tracks benchmarked against each other: a PPO baseline trained from scratch, and AMP trained against a motion prior retargeted from real human mocap. The retargeting pipeline follows [ADHERENT](reference-paper.pdf) (arXiv:2309.12784) — direct joint-space mapping from BVH to the robot's URDF, no inverse kinematics.

<p align="center">
  <img src="assets/docs/ironcub_mujoco.png" width="480" alt="iRonCub-Mk1_1 standing in MuJoCo">
</p>

## Tracks

**Track 1 — PPO (MuJoCo).** Learn to walk from scratch, no demonstration data. Trained: 2M steps, checkpoints and tensorboard logs in-repo under `checkpoints/ppo_ironcub/` and `logs/ppo_ironcub/`. This is the baseline every AMP result is compared against.

**Track 2 — AMP (Isaac Lab, SKRL).** Retarget CMU mocap into a motion prior, then train a discriminator-guided policy so gait style is shaped by the reference motion rather than emerging from the task reward alone. Env, discriminator, and trainer are written (`isaac_lab_track/`); moving to a 4090 to run the URDF→USD conversion, the stand test, and the actual training run.

The open question: does conditioning on a real human motion prior get iRonCub to a better gait, faster, than PPO gets to on its own. No conclusion yet — this updates as the AMP run progresses.

## Setup

```bash
conda create -n retarget python=3.10
conda activate retarget
conda install -c robotology idyntree   # must be conda, not pip — pip build lacks IPOPT
pip install mujoco==3.3.0 bvh scipy numpy stable-baselines3 gymnasium
```

Meshes are tracked in the repo — clone and go, no separate asset download.

## Track 1 — PPO (MuJoCo)

```bash
python mujoco_track/train_ppo.py
python mujoco_track/train_ppo.py --timesteps 3000000 --n-envs 4
python mujoco_track/train_ppo.py --resume logs/ppo_ironcub/best_model.zip

python mujoco_track/infer_ppo.py logs/ppo_ironcub/best_model.zip     # visualize policy
python mujoco_track/visualize_retargeted.py                          # visualize motion prior
```

## Track 2 — AMP (Isaac Lab)

1. **Convert URDF → USD** (once per machine, requires Isaac Lab):
   ```bash
   python {ISAACLAB}/scripts/tools/convert_urdf.py \
       assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf \
       assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd \
       --merge-fixed-joints
   ```
   `ironcub.usd` is generated locally, not tracked in git.

2. **Replay the motion prior** on the converted asset, before training:
   ```bash
   python isaac_lab_track/visualize_retargeted.py
   python isaac_lab_track/visualize_retargeted.py --headless
   ```

3. **Stand test** — sanity-check the asset and PD gains hold a stable pose:
   ```bash
   python isaac_lab_track/hardcode_stand.py
   python isaac_lab_track/hardcode_stand.py --headless --duration 20.0
   ```

4. **Train**:
   ```bash
   python isaac_lab_track/train_amp_ppo.py --headless --num_envs 4096
   python isaac_lab_track/train_amp_ppo.py --checkpoint logs/amp_ironcub/agent_50000.pt
   ```
   Logs → `logs/amp_ironcub/`.

## Regenerating motion priors

Already done for the walking clip in `motion_priors/`; for reference or a new clip:

```bash
python retargeting/retarget_bvh_npy.py \
    motion_priors/walking/07_12.bvh \
    motion_priors/walking/07_12_retargeted_adherent.npy \
    configs/ironcub_bvh_mapping.yaml

python retargeting/npy_to_npz.py \
    motion_priors/walking/07_12_retargeted_adherent.npy \
    motion_priors/walking/07_12_isaaclab.npz
```

## Retargeting — design notes

**No IK.** iRonCub has 26 DOFs; with only 2 foot position targets, IPOPT finds degenerate solutions. BVH Euler angles are mapped directly to robot joints instead, using URDF axis analysis.

**Coordinate transform.** CMU BVH is Y-up, walk direction +Z. iRonCub world is Z-up, forward +X.
```
BVH2W = [[0,0,1], [1,0,0], [0,1,0]]
```

**Sign conventions:**

| BVH channel | Robot joint | Sign | Reason |
|---|---|---|---|
| UpLeg Xrotation | hip_pitch | −1 | URDF axis [0,−1,0] |
| Leg Xrotation | knee | −1 | URDF axis [0,−1,0] |
| Foot Xrotation | ankle_pitch | +1 | URDF axis [0,+1,0] |

**MuJoCo loading.** Always load `model_stl.urdf` via `MjSpec.from_file()` with `meshdir` set. `model.xml` has an Rx(π) bug on leg bodies and is deleted from the repo.

## Repo structure

```
ampmove/
├── assets/iRonCub/
│   ├── meshes/stl/, meshes/obj/           # tracked in repo
│   └── robots/iRonCub-Mk1_1/
│       ├── model.urdf                     # iDynTree FK (retargeting)
│       ├── model_stl.urdf                 # MuJoCo + Isaac Lab USD conversion
│       └── ironcub.usd                    # generated per-machine
│
├── configs/ironcub_bvh_mapping.yaml
│
├── motion_priors/walking/
│   ├── 07_12.bvh                          # source CMU mocap (walking, subject 07)
│   ├── 07_12_retargeted_adherent.npy      # retargeted motion (root pose + joint angles)
│   └── 07_12_isaaclab.npz                 # Isaac Lab AMP input (FK link states + contacts)
│
├── retargeting/
│   ├── retarget_bvh_npy.py                # BVH → .npy
│   └── npy_to_npz.py                      # .npy → .npz (FK states + contacts for AMP)
│
├── mujoco_track/
│   ├── env.py                             # gymnasium env (26-DOF, position control)
│   ├── visualize_retargeted.py
│   ├── train_ppo.py                       # SB3 PPO
│   └── infer_ppo.py
│
└── isaac_lab_track/
    ├── ironcub_cfg.py                     # ArticulationCfg
    ├── visualize_retargeted.py
    ├── hardcode_stand.py                  # PD stand test
    ├── amp_env.py                         # DirectRLEnv: policy/AMP obs, reward, RSI reset
    └── train_amp_ppo.py                   # SKRL AMP+PPO trainer
```
