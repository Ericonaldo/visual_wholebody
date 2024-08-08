import numpy as np
import os
import torch
import cv2
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict
import wandb

from .b1z1_base import B1Z1Base, reindex_all, reindex_feet, LIN_VEL_X_CLIP, ANG_VEL_YAW_CLIP, torch_rand_int
from utils.low_level_model import ActorCritic

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
import torchvision.transforms as transforms

class B1Z1PickMulti(B1Z1Base):
    def __init__(self, table_height=None, *args, **kwargs):
        self.num_actors = 3
        super().__init__(*args, **kwargs)
        self.near_goal_stop = self.cfg["env"].get("near_goal_stop", False)
        self.obj_move_prob = self.cfg["env"].get("obj_move_prob", 0.0)
        self.table_heights_fix = table_height

    def update_roboinfo(self):
        super().update_roboinfo()
        base_obj_dis = self._cube_root_states[:, :2] - self.arm_base[:, :2]
        self.base_obj_dis = torch.norm(base_obj_dis, dim=-1)
        
    def _setup_obs_and_action_info(self):
        super()._setup_obs_and_action_info(removed_dim=9, num_action=9, num_obs=38+self.num_features-1)
        
    def _extra_env_settings(self):
        self.multi_obj = self.cfg["env"]["asset"]["asset_multi"]
        self.obj_list = list(self.multi_obj.keys())
        self.obj_height = [self.multi_obj[obj]["height"] for obj in self.obj_list]
        self.obj_orn = [self.multi_obj[obj]["orientation"] for obj in self.obj_list]
        self.obj_scale = [self.multi_obj[obj]["scale"] for obj in self.obj_list]
        obj_dir = os.path.join(self.cfg["env"]["asset"]["assetRoot"], self.cfg["env"]["asset"]["assetFileObj"])
        if not self.no_feature:
            features = []
            for obj_name in self.obj_list:
                feature_path = os.path.join(obj_dir, obj_name, "features.npy")
                feature = np.load(feature_path, allow_pickle=True)
                features.append(feature)
            assert len(features) == len(self.obj_list)
            self.features = np.concatenate(features, axis=0)
            self.num_features = self.features.shape[1]
        else:
            self.num_features = 0

    def _init_tensors(self):
        """Add extra tensors for cube and table
        """
        super()._init_tensors()
        
        self._table_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 1, :]
        self._initial_table_root_states = self._table_root_states.clone()
        self._initial_table_root_states[:, 7:13] = 0.0
        
        self._cube_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 2, :]
        self._initial_cube_root_states = self._cube_root_states.clone()
        self._initial_cube_root_states[:, 7:13] = 0.0
        
        self._table_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 1
        self._cube_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32) + 2
        
        self.table_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.table_handles[0], "box", gymapi.DOMAIN_ENV)
        
        self.lifted_success_threshold = self.cfg["env"]["liftedSuccessThreshold"]
        self.lifted_init_threshold = self.cfg["env"]["liftedInitThreshold"]
        self.base_object_distace_threshold = self.cfg["env"]["baseObjectDisThreshold"]
        self.lifted_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        
        self.highest_object = -torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.curr_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.random_angle = torch.as_tensor(np.array([0, 1.5708, -1.5708]), device=self.device, dtype=torch.float)
        self.lifted_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.pred_success:
            self.predlift_success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _create_extra(self, env_i):
        env_ptr = self.envs[env_i]
        col_group = env_i
        col_filter = 0
        i = env_i
        
        table_pos = [0.0, 0.0, self.table_dims.z / 2]
        self.table_heights[i] = table_pos[-1] + self.table_dims.z / 2

        obj_idx = i % len(self.obj_list)
        obj_asset = self.ycb_asset_list[obj_idx]
        obj_height = self.obj_height[obj_idx]
        
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(env_ptr, self.table_asset, table_start_pose, "table", col_group, col_filter, 1)
        
        cube_start_pose = gymapi.Transform()
        cube_start_pose.p.x = table_start_pose.p.x + np.random.uniform(-0.1, 0.1)
        cube_start_pose.p.y = table_start_pose.p.y + np.random.uniform(-0.1, 0.1)
        cube_start_pose.p.z = self.table_heights[i] + obj_height
        # cube_start_pose.r = quat_mul(gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-np.pi, np.pi)), gymapi.Quat(*self.obj_orn[obj_idx]))
        cube_handle = self.gym.create_actor(env_ptr, obj_asset, cube_start_pose, "cube", col_group, col_filter, 2)

        if not self.no_feature:
            self.feature_obs[i, :] = self.features[obj_idx, :]
        self.init_height[i] = obj_height
        self.init_quat[i, :] = torch.tensor(self.obj_orn[obj_idx], device=self.device)
        
        self.table_handles.append(table_handle)
        self.cube_handles.append(cube_handle)
        
    def _create_envs(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file_ycb = self.cfg["env"]["asset"]["assetFileObj"]
        
        self.table_heights = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # table
        self.table_dimz = 0.25
        self.table_dims = gymapi.Vec3(0.6, 1.0, self.table_dimz)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_options)
        table_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.table_asset)
        table_rigid_shape_props[0].friction = 0.5
        self.gym.set_asset_rigid_shape_properties(self.table_asset, table_rigid_shape_props)
        
        # cube
        ycb_opts = gymapi.AssetOptions()
        ycb_opts.use_mesh_materials = True
        ycb_opts.vhacd_enabled = True
        ycb_opts.override_inertia = True
        ycb_opts.override_com = True
        ycb_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        # ycb_opts.vhacd_params = gymapi.VhacdParams()
        # ycb_opts.vhacd_params.resolution = 500000
        self.ycb_asset_list = []
        for i in range(len(self.obj_list)):
            file_path = asset_file_ycb + self.obj_list[i] + "/model.urdf"
            ycb_asset = self.gym.load_asset(self.sim, asset_root, file_path, ycb_opts)
            ycb_asset_props = self.gym.get_asset_rigid_shape_properties(ycb_asset)
            ycb_asset_props[0].friction = 2.0
            self.gym.set_asset_rigid_shape_properties(ycb_asset, ycb_asset_props)
            self.ycb_asset_list.append(ycb_asset)

        self.table_handles, self.cube_handles = [], []
        
        if not self.no_feature:
            self.feature_obs = torch.zeros(self.num_envs, self.num_features, device=self.device, dtype=torch.float)
            self.features = torch.tensor(self.features, device=self.device, dtype=torch.float)
        self.init_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.init_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)

        super()._create_envs()
            
    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            # bowl_indices = torch.tensor([0, 9, 27, 31], device=self.device)
            # ball_indices = torch.tensor([3, 15, 17, 23], device=self.device)
            # long_box_indices = torch.tensor([1], device=self.device)
            # square_box_indices = torch.tensor([11, 12, 24], device=self.device)
            # bottle_indices = torch.tensor([2, 13, 16, 20], device=self.device)
            # cup_indices = torch.tensor([5, 28, 29], device=self.device)
            # drill_indices = torch.tensor([7], device=self.device)
            num_group = self.num_envs // 33
            bowl_indices_np = np.array([[0+i*33, 9+i*33, 27+i*33, 31+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            bowl_indices = torch.from_numpy(bowl_indices_np).to(self.device)
            ball_indices_np = np.array([[3+i*33, 15+i*33, 17+i*33, 23+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            ball_indices = torch.from_numpy(ball_indices_np).to(self.device)
            long_box_indices_np = np.array([[1+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            long_box_indices = torch.from_numpy(long_box_indices_np).to(self.device)
            square_box_indices_np = np.array([[11+i*33, 12+i*33, 24+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            square_box_indices = torch.from_numpy(square_box_indices_np).to(self.device)
            bottle_indices_np = np.array([[2+i*33, 13+i*33, 16+i*33, 20+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            bottle_indices = torch.from_numpy(bottle_indices_np).to(self.device)
            cup_indices_np = np.array([[5+i*33, 28+i*33, 29+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            cup_indices = torch.from_numpy(cup_indices_np).to(self.device)
            drill_indices_np = np.array([[7+i*33] for i in range(num_group)]).reshape(1,-1).squeeze()
            drill_indices = torch.from_numpy(drill_indices_np).to(self.device)
            
            bowl_success_time = self.success_counter[bowl_indices].sum().item(), self.episode_counter[bowl_indices].sum().item()
            ball_success_time = self.success_counter[ball_indices].sum().item(), self.episode_counter[ball_indices].sum().item()
            longbox_success_time = self.success_counter[long_box_indices].sum().item(), self.episode_counter[long_box_indices].sum().item()
            squarebox_success_time = self.success_counter[square_box_indices].sum().item(), self.episode_counter[square_box_indices].sum().item()
            bottle_success_time = self.success_counter[bottle_indices].sum().item(), self.episode_counter[bottle_indices].sum().item()
            cup_success_time = self.success_counter[cup_indices].sum().item(), self.episode_counter[cup_indices].sum().item()
            drill_success_time = self.success_counter[drill_indices].sum().item(), self.episode_counter[drill_indices].sum().item()
            
            bowl_success_rate = min(bowl_success_time[0], bowl_success_time[1]) / max(bowl_success_time[1], 1)
            ball_success_rate = min(ball_success_time[0], ball_success_time[1]) / max(ball_success_time[1], 1)
            longbox_success_rate = min(longbox_success_time[0], longbox_success_time[1]) / max(longbox_success_time[1], 1)
            squarebox_success_rate = min(squarebox_success_time[0], squarebox_success_time[1]) / max(squarebox_success_time[1], 1)
            bottle_success_rate = min(bottle_success_time[0], bottle_success_time[1]) / max(bottle_success_time[1], 1)
            cup_success_rate = min(cup_success_time[0], cup_success_time[1]) / max(cup_success_time[1], 1)
            drill_success_rate = min(drill_success_time[0], drill_success_time[1]) / max(drill_success_time[1], 1)

            wandb_dict = {
                "success_rate": {
                    "SuccessRate / Bowl": bowl_success_rate,
                    "SuccessRate / Ball": ball_success_rate,
                    "SuccessRate / LongBox": longbox_success_rate,
                    "SuccessRate / SquareBox": squarebox_success_rate,
                    "SuccessRate / Bottle": bottle_success_rate,
                    "SuccessRate / Cup": cup_success_rate,
                    "SuccessRate / Drill": drill_success_rate,
                }
            }
            if self.pred_success:
                predlift_success_rate = 0 if self.global_step_counter==0 else (self.predlift_success_counter / self.local_step_counter).mean().item()
                wandb_dict["success_rate"]["SuccessRate / PredLifted"] = predlift_success_rate
            
            if self.cfg["env"]["wandb"]:
                self.extras.update(wandb_dict)
                # wandb.log(wandb_dict, step=self.global_step_counter)
            else:
                print(wandb_dict)
                print("Bowl count: {}\n, Ball count: {}\n, LongBox count: {}\n, SquareBox count: {}\n, Bottle count: {}\n, Cup count: {}\n, Drill count: {}\n".format(bowl_success_time[1], ball_success_time[1], longbox_success_time[1], squarebox_success_time[1], bottle_success_time[1], cup_success_time[1], drill_success_time[1]))
                success_time = self.success_counter.sum().item(), self.episode_counter.sum().item()
                success_rate = min(success_time[0], success_time[1]) / max(success_time[1], 1)
                print("Total success rate", success_rate)
                
    def _reset_objs(self, env_ids):
        if len(env_ids)==0:
            return
        
        # self._cube_root_states[env_ids] = self._initial_cube_root_states[env_ids]
        self._cube_root_states[env_ids, 0] = 0.0
        self._cube_root_states[env_ids, 0] += torch_rand_float(-0.15, 0.15, (len(env_ids), 1), device=self.device).squeeze(1)
        self._cube_root_states[env_ids, 1] = 0.0
        self._cube_root_states[env_ids, 1] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).squeeze(1)
        
        self._cube_root_states[env_ids, 2] = self.table_heights[env_ids] + self.init_height[env_ids]
        rand_yaw_box = torch_rand_float(-3.15, 3.15, (len(env_ids), 1), device=self.device).squeeze(1)
        
        if True: # self.global_step_counter < 25000:
            self._cube_root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(0*rand_yaw_box, 0*rand_yaw_box, rand_yaw_box), self.init_quat[env_ids]) # Make sure to learn basic grasp
        else:
            rand_r_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            rand_p_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            self._cube_root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(rand_r_box, rand_p_box, rand_yaw_box), self.init_quat[env_ids])
        self._cube_root_states[env_ids, 7:13] = 0.
        
    def _reset_table(self, env_ids):
        if len(env_ids)==0:
            return
        
        self._table_root_states[env_ids] = self._initial_table_root_states[env_ids]
        if self.table_heights_fix is None:
            rand_heights = torch_rand_float(0, 0.5, (len(env_ids), 1), device=self.device)
        else:
            rand_heights = torch.ones((len(env_ids), 1), device=self.device, dtype=torch.float)*self.table_heights_fix - self.table_dimz / 2
        
        self._table_root_states[env_ids, 2] = rand_heights.squeeze(1) - self.table_dimz / 2.0
        self.table_heights[env_ids] = self._table_root_states[env_ids, 2] + self.table_dimz / 2.0
    
    def _reset_actors(self, env_ids):
        self._reset_table(env_ids)
        self._reset_objs(env_ids)
        super()._reset_actors(env_ids)

        return
    
    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        robot_ids_int32 = self._robot_actor_ids[env_ids]
        table_ids_int32 = self._table_actor_ids[env_ids]
        cube_ids_int32 = self._cube_actor_ids[env_ids]
        multi_ids_int32 = torch.cat([robot_ids_int32, table_ids_int32, cube_ids_int32], dim=0)
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(robot_ids_int32), len(robot_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(multi_ids_int32), len(multi_ids_int32))

        self.lifted_object[env_ids] = 0
        self.curr_height[env_ids] = 0.
        self.highest_object[env_ids] = -1.

        return
    
    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        self._update_curr_dist()
    
    def _update_curr_dist(self):
        d = torch.norm(self.ee_pos - self._cube_root_states[:, :3], dim=-1)
        self.curr_dist[:] = d
        self.closest_dist = torch.where(self.closest_dist < 0, self.curr_dist, self.closest_dist)
        
        self.curr_height[:] = self._cube_root_states[:, 2] - self.table_heights - self.init_height
        self.highest_object = torch.where(self.highest_object < 0, self.curr_height, self.highest_object)
        
    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            
        obs = super()._compute_observations(env_ids)
        
        if not self.no_feature:
            if self.cfg["env"].get("lastCommands", False):
                self.obs_buf[env_ids] = torch.cat([self.feature_obs[env_ids, :], obs, self.command_history_buf[env_ids, -1]], dim=-1)
            else:
                self.obs_buf[env_ids] = torch.cat([self.feature_obs[env_ids, :], obs, self.action_history_buf[env_ids, -1]], dim=-1)
        else:
            if self.cfg["env"].get("lastCommands", False):
                self.obs_buf[env_ids] = torch.cat([obs, self.command_history_buf[env_ids, -1]], dim=-1)
            else:
                self.obs_buf[env_ids] = torch.cat([obs, self.action_history_buf[env_ids, -1]], dim=-1)
    
    def _compute_robot_obs(self, env_ids=None):
        if env_ids is None:
            robot_root_state = self._robot_root_states
            table_root_state = self._table_root_states
            cube_root_state = self._cube_root_states
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            commands = self.commands
            table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(self.num_envs, 1).to(self.device)
            base_quat_yaw = self.base_yaw_quat
            spherical_center = self.get_ee_goal_spherical_center()
            ee_goal_cart = self.curr_ee_goal_cart
            ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy
        else:
            robot_root_state = self._robot_root_states[env_ids]
            table_root_state = self._table_root_states[env_ids]
            cube_root_state = self._cube_root_states[env_ids]
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            commands = self.commands[env_ids]
            table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(len(env_ids), 1).to(self.device)
            base_quat_yaw = self.base_yaw_quat[env_ids]
            spherical_center = self.get_ee_goal_spherical_center()[env_ids]
            ee_goal_cart = self.curr_ee_goal_cart[env_ids]
            ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy[env_ids]
        
        obs = compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos,
                                         body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, self.gripper_idx, table_dim,
                                         ee_goal_cart, ee_goal_orn_rpy, self.use_roboinfo, self.floating_base)
        
        return obs
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        cmd_dim = 2
        pred_dim = 1 if self.pred_success else 0
        if self.near_goal_stop:
            self.extras["replaced_action"] = torch.clone(actions)
            self.extras["replaced_action"][self.base_obj_dis < 0.6, -(cmd_dim+pred_dim):-pred_dim] = 0.0 # enforced these cmd to be 0
            # if not self.enable_camera:
            actions = self.extras["replaced_action"]
        
        # Randomly change the object position in a small probability (like 0.1)
        obj_move_prob = torch_rand_float(0, 1, (self.num_envs, 1), device=self.device).squeeze()
        changed_env_ids = torch.range(0, self.num_envs-1, dtype=int, device=self.device)[obj_move_prob < self.obj_move_prob]
        self._reset_objs(changed_env_ids)

        self.extras["lifted_now"] = self.lifted_now.unsqueeze(-1)*2-1 # This is for the lifted results from the last step, exactly what we want. Lifted = 1, unlifted = -1
        res = super().step(actions)

        pred_lift = actions[...,-1] > 0
        if self.pred_success and (actions.shape[-1] == (self.action_space.shape[-1]+1)):
            pred_true = (self.lifted_now == pred_lift)
            self.predlift_success_counter = torch.where(pred_true, self.predlift_success_counter + 1, self.predlift_success_counter)

        return res
    
    def check_termination(self):
        super().check_termination()

        # Check if lifted
        cube_height = self._cube_root_states[:, 2]
        box_pos = self._cube_root_states[:, :3]
        d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
        self.lifted_now = torch.logical_and((cube_height - self.table_heights) > (0.03 / 2 + self.lifted_success_threshold), d1 < 0.1)
        self.reset_buf = torch.where(~self.lifted_now & self.lifted_object, torch.ones_like(self.reset_buf), self.reset_buf) # reset the dropped envs
        self.lifted_object = torch.logical_and((cube_height - self.table_heights - self.init_height) > (self.lifted_success_threshold), d1 < 0.1)

        z_cube = self._cube_root_states[:, 2]
        # cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))
        cube_falls = z_cube < self.table_heights # Fall or model glitch
        self.reset_buf[:] = self.reset_buf | cube_falls
        # print("cube falls", cube_falls[0])
        
        if self.enable_camera:
            robot_head_dir = quat_apply(self.base_yaw_quat, torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1))
            cube_dir = self._cube_root_states[:, :3] - self._robot_root_states[:, :3]
            cube_dir[:, 2] = 0
            cube_dir = cube_dir / torch.norm(cube_dir, dim=-1).unsqueeze(-1)
            # check if dot product is negative
            deviate_much = torch.sum(robot_head_dir * cube_dir, dim=-1) < 0.
            # print("deviate much", deviate_much[0])
            
            fov_camera_pos = self._robot_root_states[:, :3] + quat_apply(self._robot_root_states[:, 3:7], torch.tensor(self.cfg["sensor"]["onboard_camera"]["position"], device=self.device).repeat(self.num_envs, 1))
            too_close_table = (fov_camera_pos[:, 0] > 0.)
            # print("too_close_table", too_close_table[0])
            
            self.reset_buf = self.reset_buf | deviate_much | too_close_table

    # --------------------------------- reward functions ---------------------------------
    def _reward_standpick(self):
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        reward[(self.base_obj_dis < self.base_object_distace_threshold) & (self.commands[:, 0] < LIN_VEL_X_CLIP)] = 1.0
        
        if self.global_step_counter < 30000:
            reward = 0.
            
        return reward, reward
    
    def _reward_grasp_base_height(self):
        cube_height = self._cube_root_states[:, 2]
        box_pos = self._cube_root_states[:, :3]
        d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
        
        reward, _ = self._reward_base_height()
        reward *= self.lifted_now # no reward for not grasped
        
        return reward, reward
    
    def _reward_approaching(self):
        """Change the reward function to be effective only when the object is lifted
        """
        reward, _ = super()._reward_approaching()
        reward *= ~self.lifted_object
        return reward, reward
    
    def _reward_base_approaching(self):
        rew, _ = super()._reward_base_approaching(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_command_reward(self):
        rew, _ = super()._reward_command_reward(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_command_penalty(self):
        rew, _ = super()._reward_command_penalty(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_ee_orn(self):
        rew, _ = super()._reward_ee_orn(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_base_dir(self):
        rew, _ = super()._reward_base_dir(self._cube_root_states[:, :3])
        return rew, rew
    
    # --------------------------------- reward functions ---------------------------------

# --------------------------------- jit functions ---------------------------------

@torch.jit.script
def compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos, 
                               body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, gripper_idx, table_dim, 
                               ee_goal_cart, ee_goal_orn_rpy, use_roboinfo, floating_base):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    cube_pos = cube_root_state[:, :3]
    cube_orn = cube_root_state[:, 3:7]
    
    ee_pos = body_pos[..., gripper_idx, :]
    ee_rot = body_rot[..., gripper_idx, :]
    ee_vel = body_vel[..., gripper_idx, :]
    ee_ang_vel = body_ang_vel[..., gripper_idx, :]
    # box pos and orientation  3+4=7
    # dof pos + vel  6+6=12
    # ee state  13
    if use_roboinfo:
        dof_pos = dof_pos[..., :]
        dof_vel = dof_vel[..., :-1] * 0.05
        if not floating_base:
            dof_pos = reindex_all(dof_pos)
            dof_vel = reindex_all(dof_vel)
    else:
        dof_pos = dof_pos[..., 12:-1] # arm_dof_pos
        dof_vel = dof_vel[..., 12:-1] # arm_dof_vel
    
    base_quat = robot_root_state[:, 3:7]
    arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=robot_root_state.device).repeat(robot_root_state.shape[0], 1)
    arm_base = quat_apply(base_quat, arm_base_local) + robot_root_state[:, :3]
    
    # cube_pos_local = quat_rotate_inverse(base_quat, cube_pos - arm_base)
    # cube_orn_local = quat_mul(quat_conjugate(base_quat), cube_orn)
    # cube_pos_local = quat_rotate_inverse(base_quat_yaw, cube_pos - spherical_center)
    cube_pos_local = quat_rotate_inverse(base_quat_yaw, cube_pos - arm_base)
    cube_pos_local[:, 2] = cube_pos[:, 2]
    cube_orn_local = quat_mul(quat_conjugate(base_quat_yaw), cube_orn)
    cube_orn_local_rpy = torch.stack(euler_from_quat(cube_orn_local), dim=-1)
    
    table_pos_local = quat_rotate_inverse(base_quat_yaw, table_root_state[:, :3] - spherical_center)
    table_orn_local = quat_mul(quat_conjugate(base_quat_yaw), table_root_state[:, 3:7])
    table_dim_local = quat_rotate_inverse(base_quat_yaw, table_dim)
    
    # ee_pos_local = quat_rotate_inverse(base_quat_yaw, ee_pos - spherical_center)
    # ee_rot_local = quat_mul(quat_conjugate(base_quat_yaw), ee_rot)
    ee_pos_local = quat_rotate_inverse(base_quat, ee_pos - arm_base)
    ee_rot_local = quat_mul(quat_conjugate(base_quat), ee_rot)
    ee_rot_local_rpy = torch.stack(euler_from_quat(ee_rot_local), dim=-1)
    
    robot_vel_local = quat_rotate_inverse(base_quat_yaw, robot_root_state[:, 7:10])
    
    # dim 3 + 3 + 3 + 3 + 7 + 7 + 3 + 3 = 32
    # 32 + 6 = 38
    # 38 + 24 = 62
    obs = torch.cat((cube_pos_local, cube_orn_local_rpy, ee_pos_local, ee_rot_local_rpy, dof_pos, dof_vel,
                     commands, ee_goal_cart, ee_goal_orn_rpy, robot_vel_local), dim=-1)
    
    return obs
