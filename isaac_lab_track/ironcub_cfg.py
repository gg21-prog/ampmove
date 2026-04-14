"""
ArticulationCfg for iRonCub-Mk1_1 in Isaac Lab.

Requires the USD asset to be generated first (run once per machine):
    python {ISAACLAB}/scripts/tools/convert_urdf.py \
        assets/iRonCub/robots/iRonCub-Mk1_1/model_stl.urdf \
        assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd \
        --merge-fixed-joints
"""

from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

USD_PATH = str(
    Path(__file__).resolve().parent.parent
    / "assets/iRonCub/robots/iRonCub-Mk1_1/ironcub.usd"
)

IRONCUB_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.57),
        rot=(1.0, 0.0, 0.0, 0.0),  # wxyz — upright
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*", ".*ankle.*"],
            stiffness=80.0,
            damping=8.0,
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=["torso.*"],
            stiffness=80.0,
            damping=8.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*shoulder.*", ".*elbow.*"],
            stiffness=80.0,
            damping=8.0,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=["neck.*"],
            stiffness=40.0,
            damping=4.0,
        ),
    },
)
