from .vec_task import VecTask
import torch
import torch.nn.functional as F
import numpy as np
from isaacgym.torch_utils import *

class RewardVecTask(VecTask):
    # --------------------------------- reward functions ---------------------------------
    def _reward_approaching(self):
        # Change the reward function to be based on the distance between the gripper and the cube
        dist_delta = self.closest_dist - self.curr_dist
        self.closest_dist = torch.minimum(self.closest_dist, self.curr_dist)
        dist_delta = torch.clip(dist_delta, 0., 10.)
        reward = torch.tanh(10.0 * dist_delta)
        # reward[(self.base_obj_dis >= self.base_object_distace_threshold)] = 0
        
        return reward, reward
    
    def _reward_reach(self):
        # table_height = self._table_root_states[:, 2] * 2.0
        goal_dist = torch.norm(self.ee_pos - self._cube_root_states[:, :3], dim=-1)
        reached_now = goal_dist < 0.08
        reward = torch.where(reached_now, torch.ones_like(self.reset_buf, dtype=torch.float), torch.zeros_like(self.reset_buf, dtype=torch.float))
        self.reset_buf = torch.where(reached_now, torch.ones_like(self.reset_buf), self.reset_buf)
        return reward, reward
    
    def _reward_lifting(self):
        height_delta = self.curr_height - self.highest_object
        self.highest_object = torch.maximum(self.highest_object, self.curr_height)
        height_delta = torch.clip(height_delta, 0., 10.)
        lifting_rew = torch.tanh(10.0 * height_delta)
        
        reward = torch.where(self.lifted_object, torch.zeros_like(lifting_rew), lifting_rew)
        
        return reward, reward
    
    def _reward_pick_up(self):
        reward = torch.where(self.lifted_object, torch.ones_like(self.reset_buf, dtype=torch.float), torch.zeros_like(self.reset_buf, dtype=torch.float))
        
        if self.global_step_counter < 20000 or self.eval:
            self.success_counter[self.lifted_object] += 1 
            self.reset_buf = torch.where(self.lifted_object, torch.ones_like(self.reset_buf), self.reset_buf) # reset the picked envs
        else:
            self.success_counter[self.lifted_object & (self.pick_counter < 1)] += 1 # Have a bug when the pick is fail during holding
            self.pick_counter = torch.where(self.lifted_object, self.pick_counter + 1, torch.zeros_like(self.pick_counter))
            self.reset_buf = torch.where(self.pick_counter >= self.hold_steps, torch.ones_like(self.reset_buf), self.reset_buf) # reset the picked after hold times
        
        return reward, reward
    
    def _reward_acc_penalty(self):
        arm_vel = self._dof_vel[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints]
        last_arm_vel = self._last_dof_vel[:, -(6 + self.num_gripper_joints):-self.num_gripper_joints]
        penalty = torch.norm(arm_vel - last_arm_vel, dim=-1) / self.dt
        reward = 1 - torch.exp(-penalty)
        return reward, reward
    
    def _reward_command_reward(self, obj_pos):
        base_obj_dis = obj_pos[:, :2] - self.arm_base[:, :2]
        base_obj_dis = torch.norm(base_obj_dis, dim=-1)
        
        reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        reward[base_obj_dis < 0.6] = torch.exp(-torch.abs(self.commands[:, 0]))[base_obj_dis < 0.6]
        
        if self.global_step_counter < 30000:
            reward = 0.

        return reward, reward
    
    def _reward_command_penalty(self, obj_pos):
        # penalty = torch.where(torch.norm(self.ee_pos - self._cube_root_states[:, :3], dim=-1) < 0.1, torch.norm(self.commands[:, :3], dim=-1), \
        #     torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)) # gripper and object
        
        base_obj_dis = obj_pos[:, :2] - self.arm_base[:, :2]
        base_obj_dis = torch.norm(base_obj_dis, dim=-1)
        
        penalty = torch.where(base_obj_dis < 0.6, torch.norm(self.commands[:, :1], dim=-1), \
            torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)) # gripper and object
        
        if self.global_step_counter < 30000:
            penalty = 0.
        else:
            pass
            # cube_height = self._cube_root_states[:, 2]
            # box_pos = self._cube_root_states[:, :3]
            # d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
            # lifted_now = torch.logical_and(((cube_height - self.table_heights) > self.lifted_init_threshold), d1 < 0.1)
            # penalty[lifted_now] = 0.
            
        return penalty, penalty
    
    def _reward_action_penalty(self):
        penalty = torch.norm(self.actions, dim=-1)
        rew = 1 - torch.exp(-10.*penalty)
        return rew, penalty
    
    def _reward_gripper_rate(self):
        diff = torch.norm(self.actions[:, 6:7] - self.last_actions[:, 6:7], dim=-1)
        rew = diff # 1 - torch.exp(-10.*diff)
        if self.global_step_counter < 30000:
            rew = 0

        return rew, diff
    
    def _reward_action_rate(self):
        diff = torch.norm(self.actions[:, 7:9] - self.last_actions[:, 7:9], dim=-1)
        rew = diff # 1 - torch.exp(-10.*diff)

        return rew, diff
    
    def _reward_ee_orn(self, obj_pos):
        ee_x_dir = torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1)
        ee_x_dir_world = quat_apply(self.ee_orn, ee_x_dir)
        obj_dir = obj_pos - self.ee_pos
        obj_dist = torch.norm(obj_dir, dim=-1)
        
        far_obj = obj_dist >= 0.01
        rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        obj_dir_unit = obj_dir[far_obj] / obj_dist[far_obj].unsqueeze(-1)
        # rew[far_obj] = torch.abs(torch.abs(torch.sum(ee_x_dir_world[far_obj] * obj_dir_unit, dim=-1)) - 1)
        rew[far_obj] = F.cosine_similarity(ee_x_dir_world[far_obj], obj_dir_unit)
        
        return rew, rew
    
    def _reward_base_approaching(self, obj_pos):
        base_obj_dis = obj_pos[:, :2] - self.arm_base[:, :2]
        base_obj_dis = torch.norm(base_obj_dis, dim=-1)
        delta_dis = torch.abs(base_obj_dis - self.base_object_distace_threshold)
        reward = torch.tanh(-10*delta_dis) + 1
        
        return reward, reward
    
    def _reward_base_dir(self, obj_pos):
        base_x_dir = torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1)
        base_x_dir_world = quat_apply(self.base_yaw_quat, base_x_dir)
        obj_dir = obj_pos - self._robot_root_states[:, :3]
        obj_dir[:,:2] = 0.
        obj_dist = torch.norm(obj_dir, dim=-1)
        
        safe_dis = obj_dist >= 0.01
        obj_dir_unit = obj_dir[safe_dis] / obj_dist[safe_dis].unsqueeze(-1)
        rew = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # rew[safe_dis] = torch.abs(torch.abs(torch.sum(base_x_dir_world[safe_dis] * obj_dir_unit, dim=-1)) - 1)
        rew[safe_dis] = F.cosine_similarity(base_x_dir_world[safe_dis], obj_dir_unit)
        
        return rew, rew
    
    # def _reward_table_contact_penalty(self):
    #     table_contact_force = torch.norm(self._contact_forces[:, self.table_idx], dim=-1)
    #     reward = torch.zeros_like(self.reset_buf, device=self.device, dtype=torch.float)
    #     reward[table_contact_force > 5] = -1.
    #     return table_contact_force, table_contact_force
    
    def _reward_rad_penalty(self):
        radius = torch.norm(self.curr_ee_goal_cart, dim=-1)
        limit = 0.9
        rew = torch.exp(-torch.abs(radius - limit) / 0.15)
        return rew, rew
    
    def _reward_base_ang_pen(self):
        base_ang_vel = quat_rotate_inverse(self._robot_root_states[:, 3:7], self._robot_root_states[:, 10:13])
        rew = torch.norm(base_ang_vel, dim=-1)
        return rew, rew
    
    def _update_base_yaw_quat(self):
        base_yaw = euler_from_quat(self._robot_root_states[:, 3:7])[2]
        self.base_yaw_euler = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self._robot_root_states[:, 2].unsqueeze(1), dim=1)
        reward = torch.exp(-torch.abs(base_height - self.cfg['reward']['base_height_target']))
        return reward, base_height
