# simulate_ironcub3.py
import os
from isaacgym import gymapi, gymutil

# Acquire Gym interface
gym = gymapi.acquire_gym()

# Parse args (e.g., --graphics_device_id)
args = gymutil.parse_arguments()

# --- Simulation Setup ---
sim_params = gymapi.SimParams()
sim_params.dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

sim = gym.create_sim(
    args.compute_device_id,
    args.graphics_device_id,
    gymapi.SIM_PHYSX,
    sim_params
)

# --- Ground Plane ---
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)  # Z-up
gym.add_ground(sim, plane_params)

# --- Load URDF ---
asset_root = os.path.join(os.path.dirname(__file__), "assets")
urdf_path = "ironcub/robots/iRonCubGazeboV3_0.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False          # iRonCub is floating-base
asset_options.replace_cylinder_with_capsule = True
asset_options.collapse_fixed_joints = False
asset_options.disable_gravity = False

robot_asset = gym.load_urdf(sim, asset_root, urdf_path, asset_options)
if not robot_asset:
    raise RuntimeError("Failed to load URDF")

# --- Create Environment & Actor ---
env = gym.create_env(sim, 
                     gymapi.Vec3(-1, -1, 0), 
                     gymapi.Vec3(1, 1, 0), 
                     1)  # 1 env

# Spawn slightly above ground (~0.6m for iRonCub CoM)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.6)
pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)

actor_handle = gym.create_actor(env, robot_asset, pose, "iRonCub3", 0, 1)

# --- Viewer ---
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise RuntimeError("Failed to create viewer")

# Position camera
gym.viewer_camera_look_at(viewer, None, 
                          gymapi.Vec3(2.0, 2.0, 1.5), 
                          gymapi.Vec3(0.0, 0.0, 0.8))

# --- Simulation Loop (no control: torque = 0) ---
while not gym.query_viewer_has_closed(viewer):
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)