# ampmove

Bipedal walking for **iRonCub-Mk1_1** via two parallel tracks:
- **Track 1 — PPO baseline**: train from scratch with SB3 in MuJoCo
- **Track 2+3 — AMP**: retarget CMU mocap → motion prior → AMP training in Isaac Lab with SKRL

Based on the [ADHERENT](reference-paper.pdf) pipeline (joint-space retargeting, no IK).

---

## Status

| Step | Status |
|------|--------|
| CMU BVH → iRonCub joint angles (retargeting) | done |
| MuJoCo visualization of retargeted motion | done |
| Convert .npy → Isaac Lab .npz (FK + 6D rot + contact) | done |
| PPO gymnasium env (MuJoCo) | done |
| PPO training + inference scripts | done |
| Isaac Lab simulation (motion replay) | done |
| Isaac Lab AMP task config | pending |
| SKRL AMP training | pending |

---

## Repo structure

```
ampmove/
├── assets/iRonCub/
│   ├── meshes/stl/                        # STL meshes (tracked in repo)
│   ├── meshes/obj/                        # OBJ meshes (tracked in repo)
│   └── robots/iRonCub-Mk1_1/
│       ├── model.urdf                     # used by iDynTree (retargeting FK)
│       ├── model_stl.urdf                 # used by MuJoCo + Isaac Lab USD conversion
│       └── ironcub.usd                   # generated per-machine (see Isaac Lab setup)
│
├── configs/
│   └── ironcub_bvh_mapping.yaml
│
├── envs/
│   └── ironcub_walk_env.py                # Gymnasium env for PPO (26-DOF, position control)
│
├── isaaclab/
│   └── ironcub_cfg.py                     # ArticulationCfg for Isaac Lab
│
├── motion_priors/walking/
│   ├── 07_12.bvh                          # source CMU mocap (walking, subject 07)
│   ├── 07_12_retargeted_adherent.npy      # retargeted motion (root pose + joint angles)
│   └── 07_12_isaaclab.npz                # Isaac Lab AMP input (FK link states + contacts)
│
└── scripts/
    ├── retarget_bvh_smoothed.py           # BVH → .npy
    ├── convert_to_isaaclab.py             # .npy → .npz (FK states + contacts for AMP)
    ├── visualize_retargeted.py            # motion replay in MuJoCo viewer
    ├── simulate_isaaclab.py               # motion replay in Isaac Lab viewer
    ├── train_ppo.py                       # SB3 PPO training
    └── infer_ppo.py                       # load checkpoint + visualize
```

---

## Setup

### 1. Clone
```bash
git clone <repo-url>
cd ampmove
```

Meshes are tracked — no separate download needed.

### 2. Conda environment
```bash
conda create -n retarget python=3.10
conda activate retarget
```

### 3. Install idyntree with IPOPT
```bash
conda install -c robotology idyntree
```
Must be the conda version — pip idyntree does not include IPOPT.

### 4. Remaining dependencies
```bash
pip install mujoco==3.3.0 bvh scipy numpy stable-baselines3 gymnasium
```

---

## Track 1 — PPO (MuJoCo)

```bash
# Train
python scripts/train_ppo.py
python scripts/train_ppo.py --timesteps 3000000 --n-envs 4

# Resume
python scripts/train_ppo.py --resume logs/ppo_ironcub/best_model.zip

# Visualize
python scripts/infer_ppo.py logs/ppo_ironcub/best_model.zip
```

Checkpoints → `checkpoints/ppo_ironcub/`, best model → `logs/ppo_ironcub/best_model.zip`.

---

## Track 2+3 — AMP (Isaac Lab)

### Step 1 — Convert URDF to USD (once per machine)

Requires Isaac Lab installed. Run from repo root:

```bash
python {ISAACLAB}/scripts/tools/convert_urdf.py \
    assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf \
    assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd \
    --merge-fixed-joints
```

`ironcub.usd` is generated locally and not tracked in git.

### Step 2 — Simulate (motion replay, no RL)

```bash
python scripts/simulate_isaaclab.py
python scripts/simulate_isaaclab.py --headless
```

Replays `07_12_retargeted_adherent.npy` at 120fps — good for verifying the asset and motion look correct before training.

### Step 3 — AMP training
*Pending.*

---

## Regenerating motion priors (already done, for reference)

```bash
# BVH → retargeted .npy
python scripts/retarget_bvh_smoothed.py \
    motion_priors/walking/07_12.bvh \
    motion_priors/walking/07_12_retargeted_adherent.npy \
    configs/ironcub_bvh_mapping.yaml

# .npy → Isaac Lab .npz
python scripts/convert_to_isaaclab.py \
    motion_priors/walking/07_12_retargeted_adherent.npy \
    motion_priors/walking/07_12_isaaclab.npz
```

---

## Retargeting — design notes

**No IK.** iRonCub has 26 DOFs. With only 2 foot position targets, IPOPT finds degenerate solutions. BVH Euler angles are mapped directly to robot joints using URDF axis analysis.

**Coordinate transform.**
CMU BVH is Y-up, walk direction +Z. iRonCub world is Z-up, forward +X.
```
BVH2W = [[0,0,1], [1,0,0], [0,1,0]]
```

**Sign conventions:**
| BVH channel | Robot joint | Sign | Reason |
|---|---|---|---|
| UpLeg Xrotation | hip_pitch | −1 | URDF axis [0,−1,0] |
| Leg Xrotation | knee | −1 | URDF axis [0,−1,0] |
| Foot Xrotation | ankle_pitch | +1 | URDF axis [0,+1,0] |

**MuJoCo loading.** Always use `model_stl.urdf` via `MjSpec.from_file()` with `meshdir` set. `model.xml` has an Rx(π) bug on leg bodies and is deleted.
