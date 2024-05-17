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

@torch.jit.script
def sphere2cart(sphere_coords):
    """ Convert spherical coordinates to cartesian coordinates
    Args:
        sphere_coords (torch.Tensor): Spherical coordinates (l, pitch, yaw)
    Returns:
        cart_coords (torch.Tensor): Cartesian coordinates (x, y, z)
    """
    l = sphere_coords[:, 0]
    pitch = sphere_coords[:, 1]
    yaw = sphere_coords[:, 2]
    cart_coords = torch.zeros_like(sphere_coords)
    cart_coords[:, 0] = l * torch.cos(pitch) * torch.cos(yaw)
    cart_coords[:, 1] = l * torch.cos(pitch) * torch.sin(yaw)
    cart_coords[:, 2] = l * torch.sin(pitch)
    return cart_coords

def cart2sphere(cart_coords):
    # type: (Tensor) -> Tensor
    """ Convert cartesian coordinates to spherical coordinates
    Args:
        cart_coords (torch.Tensor): Cartesian coordinates (x, y, z)
    Returns:
        sphere_coords (torch.Tensor): Spherical coordinates (l, pitch, yaw)
    """
    sphere_coords = torch.zeros_like(cart_coords)
    xy_len = torch.norm(cart_coords[:, :2], dim=1)
    sphere_coords[:, 0] = torch.norm(cart_coords, dim=1)
    sphere_coords[:, 1] = torch.atan2(cart_coords[:, 2], xy_len)
    sphere_coords[:, 2] = torch.atan2(cart_coords[:, 1], cart_coords[:, 0])
    return sphere_coords

# Parse arguments
args = gymutil.parse_arguments(description="Franka Tensor OSC Example",
                               custom_parameters=[
                                   {"name": "--num_envs", "type": int, "default": 32, "help": "Number of environments to create"},
                                   {"name": "--controller", "type": str, "default": "osc", "help": "Controller to use for Franka. Options are {ik, osc}"}])

controller = args.controller
assert(controller in ['ik', 'osc'])

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
franka_asset_file = "urdf/widowGo1_new/urdf/widowGo1.urdf"

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
default_dof_state["pos"][:18] = franka_mids[:18]
default_dof_pos = torch.tensor(franka_mids[:18], device='cuda:0')

if controller == 'osc':
    kp = torch.tensor([50, 50, 50, 10, 10, 10], device='cuda:0')
    kv = 2 * torch.sqrt(kp)
else:
    kp = 400
    kv = 40

# set DOF control properties (except grippers)
if controller == 'osc':
    franka_dof_props["driveMode"][:6].fill(gymapi.DOF_MODE_EFFORT)
    franka_dof_props["stiffness"][:6].fill(0.0)
    franka_dof_props["damping"][:6].fill(0.0)
else:
    franka_dof_props["driveMode"][:6+12].fill(gymapi.DOF_MODE_POS)
    franka_dof_props["stiffness"][:6+12].fill(kp)
    franka_dof_props["damping"][:6+12].fill(kv)

# set DOF control properties for grippers
franka_dof_props["driveMode"][18:].fill(gymapi.DOF_MODE_POS)
franka_dof_props["stiffness"][18:].fill(800.0)
franka_dof_props["damping"][18:].fill(40.0)

# Set up the env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
spacing = 1.0
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# default franka pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 0.)
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
    hand_handle = gym.find_actor_rigid_body_handle(env, franka_handle, "wx250s/ee_gripper_link")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_orn_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # Get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, franka_handle, "wx250s/ee_gripper_link", gymapi.DOMAIN_SIM)
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
pos_des_sphr = cart2sphere(pos_des)
pos_start_sphr = pos_des_sphr.clone()
# pos_des[:, 1] = 0.3
orn_des = torch.tensor([ 0, 0, 0, 1 ], device='cuda:0').repeat((num_envs, 1))

# Prepare jacobian tensor
# For franka, tensor shape is (num_envs, 10, 6, 9)
_jacobian = gym.acquire_jacobian_tensor(sim, "franka")
jacobian = gymtorch.wrap_tensor(_jacobian)

# Jacobian entries for end effector
hand_index = gym.get_asset_rigid_body_dict(franka_asset)["wx250s/ee_gripper_link"]
j_eef = jacobian[:, hand_index - 1, :6, 12:12+6]

# Rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# DOF state tensor
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_vel = dof_states[:, 1].view(num_envs, 20, 1)[:, 12:18]
dof_pos = dof_states[:, 0].view(num_envs, 20, 1)[:, 12:18]

print('hand_idx: ', hand_index)
print('jacobian shape: ', jacobian.shape)
print('rb_states: ', rb_states.shape)


itr = 0
t = 0
while not gym.query_viewer_has_closed(viewer):

    # Randomize desired hand orientations
    # if itr % 250 == 0 and args.orn_control:
    #     orn_des = torch.rand_like(orn_des)
    #     orn_des /= torch.norm(orn_des)


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
    rand_timesteps = 200
    
    if itr % rand_timesteps == 0:
        t = 0
        # pos_des[:, 0] = init_pos[:, 0] - 0.1
        # pos_des[:, 1] = init_pos[:, 1] + math.sin(itr / 100) * 0.2
        # pos_des[:, 2] = init_pos[:, 2] + math.cos(itr / 100) * 0.2

        # pos_des[:, 0] = init_pos[:, 0] - 0.2
        # pos_des[:, 1] = - pos_des[:, 1]
        pos_start_sphr = pos_des_sphr.clone()
        pos_des_sphr = torch.zeros_like(pos_des)
        pos_des_sphr[:, 0] = torch_rand_float(0.3, 0.7, (num_envs, 1), device='cuda:0').squeeze(-1)
        pos_des_sphr[:, 1] = torch_rand_float(0.1, 2 * np.pi / 5,  (num_envs, 1), device='cuda:0').squeeze(-1)
        pos_des_sphr[:, 2] = torch_rand_float(-0.5*np.pi, 0.5*np.pi, (num_envs, 1), device='cuda:0').squeeze(-1)
        # pos_des = sphere2cart(pos_des_sphr)
        # yaw = pos_des_sphr[:, 2] + torch.rand(num_envs, device="cuda:0") * 0.8 - 0.4


        # pos_des[:, 0] = ((torch.rand(num_envs, device="cuda:0")) * 0.3 + 0.1)
        # pos_des[:, 1] = (torch.rand(num_envs, device="cuda:0")) * 0.3 * (torch.randint(0, 2, (num_envs,), device="cuda:0") * 2 - 1)
        # pos_des[:, 2] = torch.rand(num_envs, device="cuda:0") * 0.3 + 0.2
        
        yaw = torch.atan2(pos_des[:, 1], pos_des[:, 0]) + torch.rand(num_envs, device="cuda:0") * 0.8 - 0.4
        roll = torch.rand(num_envs, device="cuda:0") * 1.2-0.6
        pitch = torch.rand(num_envs, device="cuda:0") * 1.2-0.6
        orn_des = quat_from_euler_xyz(roll, pitch, yaw)

        yaw_offset = torch.rand(num_envs, device="cuda:0") * 1. - 0.5
        # orn_des = torch.stack([0*torch.sin(yaw / 2), 0*torch.sin(yaw / 2), torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=1)

    if t % 1 == 0:
        pos_des = sphere2cart((pos_des_sphr - pos_start_sphr) * t / rand_timesteps + pos_start_sphr)
        yaw = torch.atan2(pos_des[:, 1], pos_des[:, 0]) + yaw_offset
        # roll = torch.rand(num_envs, device="cuda:0") * 1.0 - 0.5
        # pitch = torch.rand(num_envs, device="cuda:0") * 1.0 - 0.5
        orn_des = quat_from_euler_xyz(roll, pitch, yaw)
    t += 1
    itr += 1
    
    gym.clear_lines(viewer)
    sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
    sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 0, 1))
    for i in range(num_envs):
        sphere_pose = gymapi.Transform(gymapi.Vec3(pos_des[i, 0], pos_des[i, 1], pos_des[i, 2]), r=None)
        sphere_pose_2 = gymapi.Transform(gymapi.Vec3(pos_cur[i, 0], pos_cur[i, 1], pos_cur[i, 2]), r=None)
        gymutil.draw_lines(sphere_geom, gym, viewer, envs[i], sphere_pose)
        gymutil.draw_lines(sphere_geom_2, gym, viewer, envs[i], sphere_pose_2) 

    orn_cur /= torch.norm(orn_cur, dim=-1).unsqueeze(-1)
    orn_err = orientation_error(orn_des, orn_cur)

    pos_err = (pos_des - pos_cur)

    dpose = torch.cat([pos_err, orn_err], -1)

    if controller == 'ik':
        damping = 0.05
        j_eef_T = torch.transpose(j_eef, 1, 2)
        lmbda = torch.eye(6, device='cuda:0') * (damping ** 2)
        # print((j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda)).shape)
        # print(dpose[:, :3].unsqueeze(-1).shape)
        delta_pos = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose.unsqueeze(-1)).view(num_envs, 6)
        # u = kp * delta_pos - kv * dof_vel.squeeze(-1)
    else:
        pass

    for i in range(4):
        # Set tensor action
        if controller == 'ik':
            # new_pos = torch.cat([delta_pos, torch.zeros(num_envs, 2, device='cuda:0')], dim=-1) + dof_states[:, 0].view(num_envs, 8)

            # new_pos[]
            # torques = torch.cat([400*delta_pos - 40*dof_vel[:, :6].squeeze(-1), torch.zeros(num_envs, 2, device='cuda:0')], dim=-1)
            # print(torques)
            # gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torques))

            pos = torch.cat([torch.zeros(num_envs, 12, device='cuda:0'), delta_pos, torch.zeros(num_envs, 2, device='cuda:0')], dim=-1) + dof_states[:, 0].view(num_envs, 20)
            gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos))
        else:
            gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(torch.cat([u, torch.zeros(num_envs, 2, device='cuda:0')], dim=-1)))

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
