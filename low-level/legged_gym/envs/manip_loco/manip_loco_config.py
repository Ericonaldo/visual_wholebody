# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import numpy as np
import torch

class ManipLocoCfg( LeggedRobotCfg ):
    class goal_ee:
        num_commands = 3
        traj_time = [1, 3]
        hold_time = [0.5, 2]
        collision_upper_limits = [0.3, 0.15, 0.05 - 0.165]
        collision_lower_limits = [-0.2, -0.15, -0.35 - 0.165]
        underground_limit = -0.57
        num_collision_check_samples = 10
        command_mode = 'sphere'
        arm_induced_pitch = 0.5 # Added to -pos_p (negative goal pitch) to get default eef orn_p

        class sphere_center:
            x_offset = 0 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.53 # Relative to terrain

        # l_schedule = [0, 1]
        # p_schedule = [0, 1]
        # y_schedule = [0, 1]
        # arm_action_scale_schedule = [0, 1]
        # tracking_ee_reward_schedule = [0, 1]
        
        class ranges:
            init_pos_start = [0.15, np.pi/16, 0]
            init_pos_end = [0.4, np.pi/4, 0]
            pos_l = [0.2, 0.7]
            pos_p = [-2 * np.pi / 5, 1 * np.pi / 5]
            pos_y = [-1.4, 1.4]
            
            delta_orn_r = [-0.5, 0.5]
            delta_orn_p = [-0.5, 0.5]
            delta_orn_y = [-0.5, 0.5]
            
            final_tracking_ee_reward = 0.55
        sphere_error_scale = [1, 1, 1]#[1 / (ranges.final_pos_l[1] - ranges.final_pos_l[0]), 1 / (ranges.final_pos_p[1] - ranges.final_pos_p[0]), 1 / (ranges.final_pos_y[1] - ranges.final_pos_y[0])]
        orn_error_scale = [1, 1, 1]#[2 / np.pi, 2 / np.pi, 2 / np.pi]
            
    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3. # time before command are changed[s]

        lin_vel_x_schedule = [0, 1]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2
        ang_vel_pitch_clip = 0.5

        class ranges:
            lin_vel_x = [-0.8, 0.8] # min max [m/s]
            # lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
            # lin_vel_x = [0, 0]
            # ang_vel_yaw = [0, 0]
            
            # final_tracking_ang_vel_yaw_exp = 0.15


    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel =  1.0
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class env:
        num_envs = 6144
        num_actions = 12 + 6 #CAUTION
        num_torques = 12 + 6
        action_delay = -1  # -1 for no delay
        # num_dofs = 19
        num_gripper_joints = 2
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3 
        num_priv = 5 + 1 + 12
        history_len = 10
        num_observations = num_proprio * (history_len+1) + num_priv

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10 # episode length in seconds

        reorder_dofs = True
    
        teleop_mode = False # Overriden in teleop.py. When true, commands come from keyboard
        record_video = False
        stand_by = False

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = {} # = target angles [rad] when action = 0.0
        rand_yaw_range = np.pi/2
        origin_perturb_range = 0.5
        init_vel_perturb_range = 0.1

    class control:
        # PD Drive parameters:
        # Kp = [ 5.1876, 5.1876, 3.4584, 0.1729, 1.7292, 0.1729]
        # Kd = [ 0.4323, 0.4323, 0.0865, 0,      0.0864, 0]
        stiffness = {'joint': 20}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        adaptive_arm_gains = False
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = [0.4, 0.45, 0.45] * 2 + [0.4, 0.45, 0.45] * 2 + [2.1, 0.6, 0.6, 0, 0, 0]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        torque_supervision = False

    class asset( LeggedRobotCfg.asset ):
        file = ""
        foot_name = ""
        gripper_name = ""
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        collapse_fixed_joints = True # Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False
    
    class box:
        box_size = 0.1
        randomize_base_mass = True
        added_mass_range = [-0.001, 0.050]
        box_env_origins_x = 0
        box_env_origins_y_range = [0.1, 0.3]
        box_env_origins_z = box_size / 2 + 0.16
    
    class arm:
        init_target_ee_base = [0.2, 0.0, 0.2]
        grasp_offset = 0.08
        osc_kp = np.array([100, 100, 100, 30, 30, 30])
        osc_kd = 2 * (osc_kp ** 0.5)

    class domain_rand:
        observe_priv = True
        randomize_friction = True
        friction_range = [0.3, 3.0] # [0.5, 3.0]
        randomize_base_mass = True
        added_mass_range = [0., 15.]
        randomize_base_com = True
        added_com_range_x = [-0.15, 0.15]
        added_com_range_y = [-0.15, 0.15]
        added_com_range_z = [-0.15, 0.15]
        randomize_motor = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]
        
        randomize_rigids_after_start = True
        randomize_restitution = True
        restitution_range = [0.0, 1.0]

        randomize_gripper_mass = True
        gripper_added_mass_range = [0, 0.1]
        # randomize_arm_friction = True
        # arm_friction_range = [0.0, 0.2]
        # randomize_arm_ema = True
        # arm_ema_range = [0.05, 0.25]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.5

    class noise( LeggedRobotCfg.noise ):
        add_noise = False
  
    class rewards:
        # Scales set to 0 will still be logged (as zero reward and non-zero metric)
        # To not compute and log a given metric, set the scale to None
        class scales:
            alive = 0.5
            # walking only rewards
            tracking_lin_vel_x_l1 = 1.2
            # tracking_lin_vel_y = 0.3
            lin_vel_z = -1.0

            # Stand still only rewards
            stand_still = 0.0

            # common rewards
            feet_air_time =  1.0
            tracking_ang_vel = 0.5
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            collision = -1.
            action_rate = -0.005
            dof_pos_limits = -10.0
            delta_torques = -1.0e-7
            hip_pos = -0.1
            work = -0.003
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.001
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            base_height = 0.0
            torques = 0.0
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.55
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None
        
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.2 # tracking reward = exp(-error^2/sigma)
        tracking_ee_sigma = 1
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 0.4
        base_height_target = 0.25
        max_contact_force = 40. # forces above this value are penalized

    class viewer:
        pos = [-20, 0, 20]  # [m]
        lookat = [0, 0, -2]  # [m]

    
    class termination:
        r_threshold = 0.8
        p_threshold = 0.8
        z_threshold = 0.1

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        hf2mesh_method = "fast"  # grid or fast
        max_error = 0.1 # for fast
        horizontal_scale = 0.05 # [m] influence computation time by a lot
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        height = [0.00, 0.1] # [0.04, 0.1]
        gap_size = [0.02, 0.1]
        stepping_stone_distance = [0.02, 0.08]
        downsampled_scale = 0.075
        curriculum = False

        all_vertical = False
        no_flat = True
        
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)  # spreaded is benifitiall !
        num_cols = 20 # number of terrain cols (types)

        terrain_dict = {"smooth slope": 0., 
                        "rough slope up": 0.,
                        "rough slope down": 0.,
                        "rough stairs up": 0., 
                        "rough stairs down": 0., 
                        "discrete": 0., 
                        "stepping stones": 0.,
                        "gaps": 0., 
                        "rough flat": 1.0,
                        "pit": 0.0,
                        "wall": 0.0}
        terrain_proportions = list(terrain_dict.values())
        # trimesh only:
        slope_treshold = None # slopes above this threshold will be corrected to vertical surfaces
        origin_zero_z = False


class ManipLocoCfgPPO(LeggedRobotCfgPPO):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        continue_from_last_std = True
        init_std = [[0.8, 1.0, 1.0] * 4 + [1.0] * 6]
        actor_hidden_dims = [128]
        critic_hidden_dims = [128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        output_tanh = False

        leg_control_head_hidden_dims = [128, 128]
        arm_control_head_hidden_dims = [128, 128]

        priv_encoder_dims = [64, 20]

        num_leg_actions = 12
        num_arm_actions = 6

        adaptive_arm_gains = ManipLocoCfg.control.adaptive_arm_gains
        adaptive_arm_gains_scale = 10.0

        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.0
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 2e-4 #1.e-3 #5.e-4
        schedule = 'fixed' # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = None
        max_grad_norm = 1.
        min_policy_std = [[0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3]

        mixing_schedule=[1.0, 0, 3000] #if not RESUME else [1.0, 0, 1]
        torque_supervision = ManipLocoCfg.control.torque_supervision  #alert: also appears above
        torque_supervision_schedule=[0.0, 1000, 1000]
        adaptive_arm_gains = ManipLocoCfg.control.adaptive_arm_gains

        # dagger params
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 3000, 7000] #if not RESUME else [0, 1, 1000, 1000]

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 40000 # number of policy updates

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'manip_loco'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

