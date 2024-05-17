"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


Franka Operational Space Control
----------------
Operational Space Control of Franka robot to demonstrate Jacobian and Mass Matrix Tensor APIs
"""

from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import math
import numpy as np
import torch

import time

np.set_printoptions(suppress=True, precision=3)
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


# Parse arguments
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
                                   {"name": "--pos_control", "type": gymutil.parse_bool, "const": True, "default": True, "help": "Trace circular path in XZ plane"},
                                   {"name": "--orn_control", "type": gymutil.parse_bool, "const": True, "default": False, "help": "Send random orientation commands"}])

# Initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 1
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = args.use_gpu_pipeline

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    raise Exception("Failed to create sim")

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

# Add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

# Load franka asset
asset_root = "../../assets"
franka_asset_file = "urdf/interbotix_xsarm_descriptions/urdf/wx250.urdf"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.armature = 0.01
asset_options.disable_gravity = False
asset_options.collapse_fixed_joints = True

print("Loading asset '%s' from '%s'" % (franka_asset_file, asset_root))
franka_asset = gym.load_asset(
    sim, asset_root, franka_asset_file, asset_options)

# get joint limits and ranges for Franka
franka_dof_props = gym.get_asset_dof_properties(franka_asset)
franka_lower_limits = franka_dof_props['lower']
franka_upper_limits = franka_dof_props['upper']
franka_ranges = franka_upper_limits - franka_lower_limits
franka_mids = 0.5 * (franka_upper_limits + franka_lower_limits)
franka_num_dofs = len(franka_dof_props)
franka_num_bodies = gym.get_asset_rigid_body_count(franka_asset)

print("franka_num_dofs: ", gym.get_asset_dof_count(franka_asset), gym.get_asset_dof_names(franka_asset))
print("franka_num_bodys: ", gym.get_asset_rigid_body_count(franka_asset), gym.get_asset_rigid_body_names(franka_asset))

# set default DOF states
default_dof_state = np.zeros(franka_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"][:5] = franka_mids[:5]

# set DOF control properties (except grippers)
franka_dof_props["driveMode"][:5].fill(gymapi.DOF_MODE_EFFORT)
franka_dof_props["stiffness"][:5].fill(0.0)
franka_dof_props["damping"][:5].fill(0.0)
# franka_dof_props["driveMode"][:5].fill(gymapi.DOF_MODE_POS)
# franka_dof_props["stiffness"][:5].fill(400.0)
# franka_dof_props["damping"][:5].fill(40.0)

# set DOF control properties for grippers
franka_dof_props["driveMode"][5:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][5:].fill(800.0)
franka_dof_props["damping"][5:].fill(40.0)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default franka pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0)
pose.r = gymapi.Quat(0, 0, 0, 1)

print("Creating %d environments" % num_envs)

envs = []
hand_idxs = []
init_pos_list = []
init_orn_list = []

for i in range(num_envs):
    # Create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # Add franka
    franka_handle = gym.create_actor(env, franka_asset, pose, "franka", i, 1)

    # Set initial DOF states
    gym.set_actor_dof_states(env, franka_handle, default_dof_state, gymapi.STATE_ALL)

    # Set DOF control properties
    gym.set_actor_dof_properties(env, franka_handle, franka_dof_props)

    # Get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "wx250/ee_gripper_link")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # Get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "wx250/ee_gripper_link", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

# Point camera at middle env
cam_pos = gymapi.Vec3(4, 3, 3)
cam_target = gymapi.Vec3(-4, -3, 0)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, middle_env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API to access and control the physics simulation
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3)
init_orn = torch.Tensor(init_orn_list).view(num_envs, 4)

if args.use_gpu_pipeline:
    init_pos = init_pos.to('cuda:0')
    init_orn = init_orn.to('cuda:0')

# desired hand positions and orientations
pos_des = init_pos.clone()
pos_des[:, 1] = 0.3
orn_des = torch.tensor([ 0, 0.7071068, 0, 0.7071068 ], device='cuda:0').repeat((num_envs, 1))

# Prepare jacobian tensor
# For franka, tensor shape is (num_envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# Jacobian entries for end effector
hand_index = gym.get_asset_rigid_body_dict(franka_asset)["wx250/ee_gripper_link"]
# j_eef = jacobian[:, hand_index - 1]
j_eef = jacobian[:, hand_index - 1, :3, :5]

# Prepare mass matrix tensor
# For franka, tensor shape is (num_envs, 9, 9)
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "franka")
mm = gymtorch.wrap_tensor(_massmatrix)

kp = 50
kv = 2 * math.sqrt(kp)

# Rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# DOF state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_vel = dof_states[:, 1].view(num_envs, 7, 1)
dof_pos = dof_states[:, 0].view(num_envs, 7, 1)

print('hand_idx: ', hand_index)
print('mm shape: ', mm.shape)
print('jacobian shape: ', jacobian.shape)
print('rb_states: ', rb_states.shape)

link_mass = torch.zeros(num_envs, franka_num_bodies - 1, dtype=torch.float, device="cuda:0")
for i, env in enumerate(envs):
    rigid_body_props = gym.get_actor_rigid_body_properties(env, 0)
    for j, prop in enumerate(rigid_body_props[1:]):
        link_mass[i, j] = prop.mass

itr = 0
while not gym.query_viewer_has_closed(viewer):

    # Randomize desired hand orientations
    if itr % 250 == 0 and args.orn_control:
        orn_des = torch.rand_like(orn_des)
        orn_des /= torch.norm(orn_des)
    # orn_des = torch.tensor([ 0, 0, 0.7071068, 0.7071068 ], device="cuda:0").repeat((num_envs, 1))

    itr += 1

    # Update jacobian and mass matrix
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    # Get current hand poses
    pos_cur = rb_states[hand_idxs, :3]
    orn_cur = rb_states[hand_idxs, 3:7]
    hand_vel = rb_states[hand_idxs, 7:]

    # Set desired hand positions
    if args.pos_control and itr % 200 == 0:
        # pos_des[:, 0] = init_pos[:, 0] - 0.1
        # pos_des[:, 1] = init_pos[:, 1] + math.sin(itr / 100) * 0.2
        # pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 100) * 0.2

        # pos_des[:, 0] = init_pos[:, 0] - 0.2
        # pos_des[:, 1] = - pos_des[:, 1]

        pos_des[:, 0] = ((torch.rand(num_envs, device="cuda:0")) * 0.3 + 0.2) * (torch.randint(0, 2, (num_envs,), device="cuda:0") * 2 - 1)
        pos_des[:, 1] = (torch.rand(num_envs, device="cuda:0")) * 0.3
        pos_des[:, 2] = torch.rand(num_envs, device="cuda:0") * 0.3 + 0.25


    gym.clear_lines(viewer)
    # self.gym.refresh_rigid_body_state_tensor(self.sim)
    sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
    sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 1))
    for i in range(num_envs):
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_des[i, 0], pos_des[i, 1], pos_des[i, 2]), r=None)
        sphere_pose_2 = gymapi.Transform(gymapi.Vec3(pos_cur[i, 0], pos_cur[i, 1], pos_cur[i, 2]), r=None)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], sphere_pose)
        gymutil.draw_lines(sphere_geom_2, gym, viewer, envs[i], sphere_pose_2) 

    # Solve for control (Operational Space Control)
    m_inv = torch.inverse(mm[:, :5, :5])
    m_eef = torch.inverse(j_eef @ m_inv @ torch.transpose(j_eef, 1, 2))

    orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
    orn_err = orientation_error(orn_des, orn_cur)
    orn_err = torch.zeros_like(orn_err)

    pos_err = (pos_des - pos_cur)

    # if not args.pos_control:
    #     pos_err *= 0

    
    dpose = torch.cat([pos_err, orn_err], -1)
    # print(dpose[0].squeeze())

    damping = 0.5
    j_eef_T = torch.transpose(j_eef, 1, 2)
    # print(j_eef[0].cpu().numpy())
    lmbda = torch.eye(6, device='cuda:0') * (damping ** 2)
    # delta_pos = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)).view(num_envs, 7)
    # u = kp * delta_pos - kv * dof_vel.squeeze(-1)

    u = (torch.transpose(j_eef, 1, 2) @ m_eef @ (kp * dpose - kv * hand_vel)[:, :3].unsqueeze(-1)).view(num_envs, 5)


    if not asset_options.disable_gravity:
        g = torch.zeros(num_envs, franka_num_bodies - 1, 6, 1, dtype=torch.float, device='cuda:0')
        g[:, :, 2, :] = 9.81
        g_force = link_mass.unsqueeze(-1).unsqueeze(-1) * g

        g_torque = (torch.transpose(jacobian[:, :, :, :5], 2, 3) @ g_force).squeeze(-1)           # new shape is (n_envs, n_bodies, n_links)
        g_torque = torch.sum(g_torque, dim=1, keepdim=False)

        u += g_torque
    
    u = torch.cat([u, torch.zeros(num_envs, 2, device='cuda:0')], dim=-1)

    # time.sleep(10)
    # print('\n\nu: ', u.squeeze(-1)[0].cpu().numpy())

    # print('diff: ', u[0, :3] - dof_pos[0, :3].squeeze(-1))
    
    # Set tensor action
    # gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(delta_pos + dof_pos.squeeze(-1)))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(u))

    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Step rendering
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    # gym.sync_frame_time(sim)

print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
