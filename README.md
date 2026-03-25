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
| Isaac Lab robot asset setup (USD) | pending |
| Isaac Lab AMP task config | pending |
| SKRL AMP training | pending |

---

## Repo structure

```
ampmove/
├── assets/iRonCub/
│   ├── meshes/stl/                        # STL meshes (MuJoCo + Isaac Lab USD conversion)
│   └── robots/iRonCub-Mk1_1/
│       ├── model.urdf                     # used by iDynTree (retargeting FK)
│       └── model_stl.urdf                # used by MuJoCo (env + visualizer)
│
├── configs/
│   └── ironcub_bvh_mapping.yaml           # BVH joint → robot joint correspondence
│
├── envs/
│   └── ironcub_walk_env.py                # Gymnasium env for PPO (26-DOF, position control)
│
├── motion_priors/walking/
│   ├── 07_12.bvh                          # source CMU mocap (walking, subject 07)
│   ├── 07_12_retargeted_adherent.npy      # retargeted motion (root pose + joint angles)
│   └── 07_12_isaaclab.npz                # Isaac Lab AMP input (+ FK link states + contacts)
│
└── scripts/
    ├── retarget_bvh_smoothed.py           # BVH → .npy  (core retargeting pipeline)
    ├── convert_to_isaaclab.py             # .npy → .npz (adds FK states for AMP)
    ├── visualize_retargeted.py            # play back .npy in MuJoCo viewer
    ├── train_ppo.py                       # SB3 PPO training
    └── infer_ppo.py                       # load checkpoint + visualize
```

---

## Setup

### 1. Get the STL meshes
The mesh files are not tracked in this repo (large binaries). Get the iRonCub model package:
```bash
git clone https://github.com/icub-tech-iit/ergocub-software   # or the iRonCub model repo
# copy assets/iRonCub/meshes/stl/ into this repo at the same path
```
The URDF expects meshes at `assets/iRonCub/meshes/stl/*.stl` with absolute paths — update the paths in `model_stl.urdf` if your clone is in a different location.

### 2. Create conda environment
```bash
conda create -n retarget python=3.10
conda activate retarget
```

### 3. Install idyntree with IPOPT (required for FK in convert script)
```bash
conda install -c robotology idyntree
```
> Must be the conda version — pip idyntree does not include IPOPT support.

### 4. Install remaining dependencies
```bash
pip install mujoco==3.3.0 bvh scipy numpy stable-baselines3 gymnasium
```

---

## Track 1 — PPO (MuJoCo, local)

### Train
```bash
python scripts/train_ppo.py
# or
python scripts/train_ppo.py --timesteps 3000000 --n-envs 4
```

Checkpoints saved to `checkpoints/ppo_ironcub/`, best model to `logs/ppo_ironcub/best_model.zip`.

### Resume
```bash
python scripts/train_ppo.py --resume logs/ppo_ironcub/best_model.zip
```

### Visualize
```bash
python scripts/infer_ppo.py logs/ppo_ironcub/best_model.zip
```

---

## Track 2+3 — AMP (Isaac Lab, remote machine)

### Step 1 — Retarget (run locally, already done)
```bash
python scripts/retarget_bvh_smoothed.py \
    motion_priors/walking/07_12.bvh \
    motion_priors/walking/07_12_retargeted_adherent.npy \
    configs/ironcub_bvh_mapping.yaml
```

### Step 2 — Convert to Isaac Lab format (run locally, already done)
```bash
python scripts/convert_to_isaaclab.py \
    motion_priors/walking/07_12_retargeted_adherent.npy \
    motion_priors/walking/07_12_isaaclab.npz
```

### Step 3 — Transfer to remote machine
Copy the following to the Isaac Lab machine:
```
assets/iRonCub/              # full folder (URDF + STL meshes)
motion_priors/walking/07_12_isaaclab.npz
```

### Step 4 — Convert URDF to USD (on remote machine, Isaac Lab installed)
```bash
python scripts/tools/convert_urdf.py \
    assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf \
    assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd \
    --merge-fixed-joints
```

### Step 5 — Isaac Lab robot + AMP task setup
*Pending — see Isaac Lab track documentation once robot asset is set up.*

---

## Retargeting — key design decisions

**No IK.** iRonCub has 26 DOFs. With only 2 foot position targets, IK leaves 20 free DOFs — IPOPT finds degenerate solutions. Instead, BVH Euler angles are mapped directly to robot joints using axis analysis from the URDF.

**Coordinate transform.**
CMU BVH is Y-up, walk direction +Z. iRonCub world is Z-up, forward +X.
```
BVH2W = [[0,0,1], [1,0,0], [0,1,0]]   # Z_bvh→X_world, X_bvh→Y_world, Y_bvh→Z_world
```

**Sign conventions** (from URDF joint axis analysis):
| BVH channel | Robot joint | Sign | Reason |
|---|---|---|---|
| LeftUpLeg Xrotation | l_hip_pitch | −1 | URDF axis [0,−1,0] |
| LeftLeg Xrotation | l_knee | −1 | URDF axis [0,−1,0] |
| LeftFoot Xrotation | l_ankle_pitch | +1 | URDF axis [0,+1,0] |

**MuJoCo model note.** Load from `model_stl.urdf` via `MjSpec.from_file()` — do NOT use `model.xml` (has an Rx(π) body orientation bug on `l_upper_leg`/`r_upper_leg` that inverts the kinematic chain).
