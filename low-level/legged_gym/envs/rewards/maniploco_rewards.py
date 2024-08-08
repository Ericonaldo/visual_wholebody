import torch
from isaacgym.torch_utils import *

class ManipLoco_rewards:
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env
    # -------------Z1: Reward functions----------------

    def _reward_tracking_ee_sphere(self):
        ee_pos_local = quat_rotate_inverse(self.env.base_yaw_quat, self.env.ee_pos - self.env.get_ee_goal_spherical_center())
        ee_pos_error = torch.sum(torch.abs(cart2sphere(ee_pos_local) - self.env.curr_ee_goal_sphere) * self.env.sphere_error_scale, dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error

    def _reward_tracking_ee_world(self):
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - self.env.curr_ee_goal_cart_world), dim=1)
        rew = torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma * 2)
        return rew, ee_pos_error

    def _reward_tracking_ee_sphere_walking(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_sphere_standing(self):
        reward, metric = self.env._reward_tracking_ee_sphere()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_tracking_ee_cart(self):
        target_ee = self.env.get_ee_goal_spherical_center() + quat_apply(self.env.base_yaw_quat, self.env.curr_ee_goal_cart)
        ee_pos_error = torch.sum(torch.abs(self.env.ee_pos - target_ee), dim=1)
        return torch.exp(-ee_pos_error/self.env.cfg.rewards.tracking_ee_sigma), ee_pos_error
    
    def _reward_tracking_ee_orn(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs(torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler)) * self.env.orn_error_scale, dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    def _reward_arm_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, 12:-self.env.cfg.env.num_gripper_joints] * self.env.dof_vel[:, 12:-self.env.cfg.env.num_gripper_joints]), dim = 1)
        return energy, energy

    def _reward_tracking_ee_orn_ry(self):
        ee_orn_euler = torch.stack(euler_from_quat(self.env.ee_orn), dim=-1)
        orn_err = torch.sum(torch.abs((torch_wrap_to_pi_minuspi(self.env.ee_goal_orn_euler - ee_orn_euler) * self.env.orn_error_scale)[:, [0, 2]]), dim=1)
        return torch.exp(-orn_err/self.env.cfg.rewards.tracking_ee_sigma), orn_err

    # -------------B1: Reward functions----------------

    def _reward_hip_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, [0, 3, 6, 9]] ** 2, dim=1)
        return action_l2, action_l2

    def _reward_leg_energy_abs_sum(self):
        energy = torch.sum(torch.abs(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim = 1)
        return energy, energy

    def _reward_leg_energy_sum_abs(self):
        energy = torch.abs(torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1))
        return energy, energy
    
    def _reward_leg_action_l2(self):
        action_l2 = torch.sum(self.env.actions[:, :12] ** 2, dim=1)
        return action_l2, action_l2
    
    def _reward_leg_energy(self):
        energy = torch.sum(self.env.torques[:, :12] * self.env.dof_vel[:, :12], dim = 1)
        return energy, energy
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.env.cfg.rewards.tracking_sigma), lin_vel_error

    def _reward_tracking_lin_vel_x_l1(self):
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < 1e-5
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        rew = 0*error
        rew_x = -error + torch.abs(self.env.commands[:, 0])
        rew[~zero_cmd_indices] = rew_x[~zero_cmd_indices] / (torch.abs(self.env.commands[~zero_cmd_indices, 0]) + 0.01)
        rew[zero_cmd_indices] = 0
        return rew, error

    def _reward_tracking_lin_vel_x_exp(self):
        error = torch.abs(self.env.commands[:, 0] - self.env.base_lin_vel[:, 0])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_ang_vel_yaw_l1(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return - error + torch.abs(self.env.commands[:, 2]), error
    
    def _reward_tracking_ang_vel_yaw_exp(self):
        error = torch.abs(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-error/self.env.cfg.rewards.tracking_sigma), error

    def _reward_tracking_lin_vel_y_l2(self):
        squared_error = (self.env.commands[:, 1] - self.env.base_lin_vel[:, 1]) ** 2
        return squared_error, squared_error
    
    def _reward_tracking_lin_vel_z_l2(self):
        squared_error = (self.env.commands[:, 2] - self.env.base_lin_vel[:, 2]) ** 2
        return squared_error, squared_error
    
    def _reward_survive(self):
        survival_reward = torch.ones(self.env.num_envs, device=self.env.device)
        return survival_reward, survival_reward

    def _reward_foot_contacts_z(self):
        foot_contacts_z = torch.square(self.env.force_sensor_tensor[:, :, 2]).sum(dim=-1)
        return foot_contacts_z, foot_contacts_z

    def _reward_torques(self):
        # Penalize torques
        torque = torch.sum(torch.square(self.env.torques), dim=1)
        return torque, torque
    
    def _reward_energy_square(self):
        energy = torch.sum(torch.square(self.env.torques[:, :12] * self.env.dof_vel[:, :12]), dim=1)
        return energy, energy

    def _reward_tracking_lin_vel_y(self):
        cmd = self.env.commands[:, 1].clone()
        lin_vel_y_error = torch.square(cmd - self.env.base_lin_vel[:, 1])
        rew = torch.exp(-lin_vel_y_error/self.env.cfg.rewards.tracking_sigma)
        return rew, lin_vel_y_error
    
    def _reward_lin_vel_z(self):
        rew = torch.square(self.env.base_lin_vel[:, 2])
        return rew, rew
    
    def _reward_ang_vel_xy(self):
        rew = torch.sum(torch.square(self.env.base_ang_vel[:, :2]), dim=1)
        return rew, rew
    
    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.env.cfg.rewards.tracking_sigma), ang_vel_error
    
    def _reward_work(self):
        work = self.env.torques * self.env.dof_vel
        abs_sum_work = torch.abs(torch.sum(work[:, :12], dim = 1))
        return abs_sum_work, abs_sum_work
    
    def _reward_dof_acc(self):
        rew = torch.sum(torch.square((self.env.last_dof_vel - self.env.dof_vel)[:, :12] / self.env.dt), dim=1)
        return rew, rew
    
    def _reward_action_rate(self):
        action_rate = torch.sum(torch.square(self.env.last_actions - self.env.actions)[:, :12], dim=1)
        return action_rate, action_rate
    
    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.env.dof_pos - self.env.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.env.dof_pos - self.env.dof_pos_limits[:, 1]).clip(min=0.) # upper limit
        rew = torch.sum(out_of_limits[:, :12], dim=1)
        return rew, rew
    
    def _reward_delta_torques(self):
        rew = torch.sum(torch.square(self.env.torques - self.env.last_torques)[:, :12], dim=1)
        return rew, rew
    
    def _reward_collision(self):
        rew = torch.sum(1.*(torch.norm(self.env.contact_forces[:, self.env.penalized_contact_indices, :], dim=-1) > 0.1), dim=1)
        return rew, rew
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_walking_dof(self):
        # Penalize motion at zero commands
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        rew[~self.env._get_walking_cmd_mask()] = 0.
        return rew, rew

    def _reward_hip_pos(self):
        rew = torch.sum(torch.square(self.env.dof_pos[:, self.env.hip_indices] - self.env.default_dof_pos[self.env.hip_indices]), dim=1)
        return rew, rew

    def _reward_feet_jerk(self):
        if not hasattr(self, "last_contact_forces"):            
            result = torch.zeros(self.env.num_envs).to(self.env.device)
        else:
            result = torch.sum(torch.norm(self.env.force_sensor_tensor - self.env.last_contact_forces, dim=-1), dim=-1)
        
        self.env.last_contact_forces = self.env.force_sensor_tensor.clone()
        result[self.env.episode_length_buf<50] = 0.
        return result, result
    
    def _reward_alive(self):
        return 1., 1.
    
    def _reward_feet_drag(self):
        feet_xyz_vel = torch.abs(self.env.rigid_body_state[:, self.env.feet_indices, 7:10]).sum(dim=-1)
        dragging_vel = self.env.foot_contacts_from_sensor * feet_xyz_vel
        rew = dragging_vel.sum(dim=-1)
        return rew, rew

    def _reward_feet_contact_forces(self):
        reset_flag = (self.env.episode_length_buf > 2./self.env.dt).type(torch.float)
        forces = torch.sum((torch.norm(self.env.force_sensor_tensor, dim=-1) - self.env.cfg.rewards.max_contact_force).clip(min=0), dim=-1)
        rew = reset_flag * forces
        return rew, rew
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        error = torch.sum(torch.square(self.env.projected_gravity[:, :2]), dim=1)
        return error, error
    
    def _reward_roll(self):
        # Penalize non flat base orientation
        roll = self.env._get_body_orientation()[:, 0]
        error = torch.abs(roll)
        return error, error
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.env.root_states[:, 2].unsqueeze(1), dim=1)
        return torch.abs(base_height - self.env.cfg.rewards.base_height_target), base_height
    
    def _reward_orientation_walking(self):
        reward, metric = self.env._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric
    
    def _reward_orientation_standing(self):
        reward, metric = self.env._reward_orientation()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_torques_walking(self):
        reward, metric = self.env._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_torques_standing(self):
        reward, metric = self.env._reward_torques()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric
    
    def _reward_energy_square_walking(self):
        reward, metric = self.env._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric
    
    def _reward_energy_square_standing(self):
        reward, metric = self.env._reward_energy_square()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric

    def _reward_base_height_walking(self):
        reward, metric = self.env._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[~walking_mask] = 0
        metric[~walking_mask] = 0
        return reward, metric

    def _reward_base_height_standing(self):
        reward, metric = self.env._reward_base_height()
        walking_mask = self.env._get_walking_cmd_mask()
        reward[walking_mask] = 0
        metric[walking_mask] = 0
        return reward, metric
    
    def _reward_dof_default_pos(self):
        dof_error = torch.sum(torch.abs(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        rew = torch.exp(-dof_error*0.05)
        
        return rew, rew
    
    def _reward_dof_error(self):
        dof_error = torch.sum(torch.square(self.env.dof_pos - self.env.default_dof_pos)[:, :12], dim=1)
        return dof_error, dof_error
    
    def _reward_tracking_lin_vel_max(self):
        rew = torch.where(self.env.commands[:, 0] > 0, torch.minimum(self.env.base_lin_vel[:, 0], self.env.commands[:, 0]) / (self.env.commands[:, 0] + 1e-5), \
                          torch.minimum(-self.env.base_lin_vel[:, 0], -self.env.commands[:, 0]) / (-self.env.commands[:, 0] + 1e-5))
        zero_cmd_indices = torch.abs(self.env.commands[:, 0]) < self.env.cfg.commands.lin_vel_x_clip
        rew[zero_cmd_indices] = torch.exp(-torch.abs(self.env.base_lin_vel[:, 0]))[zero_cmd_indices]
        return rew, rew
    
    def _reward_penalty_lin_vel_y(self):
        rew = torch.abs(self.env.base_lin_vel[:, 1])
        rot_indices = torch.abs(self.env.commands[:, 2]) > self.env.cfg.commands.ang_vel_yaw_clip
        rew[rot_indices] = 0.
        return rew, rew
    
    # -------------B1 Gait Control Rewards----------------
    def _reward_tracking_contacts_shaped_force(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0,0
        foot_forces = torch.norm(self.env.contact_forces[:, self.env.feet_indices, :], dim=-1)
        desired_contact = self.env.desired_contact_states

        reward = 0
        for i in range(4):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.env.cfg.rewards.gait_force_sigma))
        
        # cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        # reward[cmd_stop_flag] = 0
        return reward / 4, reward / 4

    def _reward_tracking_contacts_shaped_vel(self):
        if not self.env.cfg.env.observe_gait_commands:
            return 0,0
        foot_velocities = torch.norm(self.env.foot_velocities, dim=2).view(self.env.num_envs, -1)
        desired_contact = self.env.desired_contact_states
        reward = 0
        for i in range(4):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.env.cfg.rewards.gait_vel_sigma)))
        # cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        # reward[cmd_stop_flag] = 0
        
        return reward / 4, reward / 4
    
    def _reward_feet_height(self):
        feet_height_tracking = self.env.cfg.rewards.feet_height_target

        if self.env.cfg.rewards.feet_height_allfeet:
            feet_height = self.env.rigid_body_state[:, self.env.feet_indices, 2] # All feet
        else:
            feet_height = self.env.rigid_body_state[:, self.env.feet_indices[:2], 2] # Only front feet

        rew = torch.clamp(torch.norm(feet_height, dim=-1) - feet_height_tracking, max=0)
        cmd_stop_flag = ~self.env._get_walking_cmd_mask()
        rew[cmd_stop_flag] = 0
        return rew, rew

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        first_contact = (self.env.feet_air_time > 0.) * self.env.foot_contacts_from_sensor  #self.env.contact_filt
        self.env.feet_air_time += self.env.dt

        if self.env.cfg.rewards.feet_aritime_allfeet:
            rew_airTime = torch.sum((self.env.feet_air_time - 0.5) * first_contact, dim=1)
        else:
            rew_airTime = torch.sum((self.env.feet_air_time[:, :2] - 0.5) * first_contact[:, :2], dim=1)
        
        rew_airTime *= self.env._get_walking_cmd_mask()  # reward for stepping for any of the 3 motions
        self.env.feet_air_time *= ~ self.env.foot_contacts_from_sensor  #self.env.contact_filt
        return rew_airTime, rew_airTime