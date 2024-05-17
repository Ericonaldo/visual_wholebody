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

from .manip_loco_config import ManipLocoCfg, ManipLocoCfgPPO
import numpy as np

class B1Z1RoughCfg( ManipLocoCfg ):
    class goal_ee ( ManipLocoCfg.goal_ee ):
        collision_upper_limits = [0.1, 0.2, -0.05]
        collision_lower_limits = [-0.8, -0.2, -0.7]
        underground_limit = -0.7
        arm_induced_pitch = 0.38 # Added to -pos_p (negative goal pitch) to get default eef orn_p
    
        class sphere_center ( ManipLocoCfg.goal_ee.sphere_center ):
            x_offset = 0.3 # Relative to base
            y_offset = 0 # Relative to base
            z_invariant_offset = 0.7 # Relative to terrain
        
        class ranges(ManipLocoCfg.goal_ee.ranges):
            init_pos_start = [0.5, np.pi/8, 0]
            init_pos_end = [0.7, 0, 0]
            pos_l = [0.4, 0.95]
            pos_p = [-1 * np.pi / 2.5, 1 * np.pi / 3]
            pos_y = [-1.2, 1.2]

    class commands:
        curriculum = True
        num_commands = 3
        resampling_time = 3. # time before command are changed[s]

        lin_vel_x_schedule = [0, 1]
        ang_vel_yaw_schedule = [0, 1]
        tracking_ang_vel_yaw_schedule = [0, 1]

        ang_vel_yaw_clip = 0.5
        lin_vel_x_clip = 0.2
        ang_vel_pitch_clip = ang_vel_yaw_clip

        class ranges:
            lin_vel_x = [-0.6, 0.6] # min max [m/s]
            # lin_vel_y = [0, 0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            ang_vel_pitch = [-1.0, 1.0]    # min max [rad/s]

    class env (ManipLocoCfg.env):
        num_gripper_joints = 1 # 2 for two finger gripper
        num_priv = 5 + 1 + 12
        history_len = 10
        num_proprio = 2 + 3 + 18 + 18 + 12 + 4 + 3 + 3 + 3
        num_observations = num_proprio * (history_len+1) + num_priv
        action_delay = 3 # Not used, assigned in code
        observe_gait_commands = False
        frequencies = 2
        observe_velocities = True
        stand_only = False

    class init_state( ManipLocoCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.2,   # [rad]
            'FL_thigh_joint': 0.8,     # [rad]
            'FL_calf_joint': -1.5,   # [rad]

            'RL_hip_joint': 0.2,   # [rad]
            'RL_thigh_joint': 0.8,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]

            'FR_hip_joint': -0.2 ,  # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'FR_calf_joint': -1.5,  # [rad]

            'RR_hip_joint': -0.2,   # [rad]
            'RR_thigh_joint': 0.8,   # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'z1_waist': 0.0,
            'z1_shoulder': 1.48,
            'z1_elbow': -1.5, # -0.63,
            'z1_wrist_angle': 0, # -0.84,
            'z1_forearm_roll': 0.0,
            'z1_wrist_rotate': 1.57, # 0.0,
            'z1_jointGripper': -0.785,
        }

    class control( ManipLocoCfg.control ):
        stiffness = {'joint': 80, 'z1': 5}  # [N*m/rad] # Kp: 80, 150, 200
        damping = {'joint': 2.0, 'z1': 0.5}     # [N*m*s/rad]

    class asset( ManipLocoCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/b1z1/urdf/b1z1.urdf'
        foot_name = "foot"
        gripper_name = "ee_gripper_link" #"gripperMover"
        # gripper_name = "wx250s/ee_gripper_link" # for two finger gripper
        penalize_contacts_on = ["thigh", "trunk", "calf"]
        terminate_after_contacts_on = []
  
    class rewards ( ManipLocoCfg.rewards ):
        gait_vel_sigma = 0.5
        gait_force_sigma = 0.5
        kappa_gait_probs = 0.07
        class scales ( ManipLocoCfg.rewards.scales ):
            tracking_contacts_shaped_force = -2.0 # Only works when `observing_gait_commands` is true
            tracking_contacts_shaped_vel = -2.0 # Only works when `observing_gait_commands` is true
            tracking_lin_vel_max = 2.0 # 1.5
            tracking_lin_vel_x_l1 = 0.
            tracking_lin_vel_x_exp = 0
            tracking_ang_vel = 0.5 # just for yaw
            delta_torques = -1.0e-7/4.0
            work = 0
            energy_square = 0.0
            torques = -2.5e-5 # -1e-5
            stand_still = 1.0 #1.5
            walking_dof = 1.5
            dof_default_pos = 0.0
            dof_error = 0.0 # -0.06 # -0.04
            alive = 1.0
            lin_vel_z = -1.5
            roll = -2.0
            
            tracking_ang_pitch_vel = 0.5 # New reward, only useful when pitch_control = True

            # common rewards
            feet_air_time = 1.0
            feet_height = 1.0
            ang_vel_xy = -0.2 # -0.1
            dof_acc = -7.5e-7 #-2.5e-7
            collision = -10.
            action_rate = -0.015
            dof_pos_limits = -10.0
            hip_pos = -0.3
            feet_jerk = -0.0002
            feet_drag = -0.08
            feet_contact_forces = -0.001
            orientation = 0.0
            orientation_walking = 0.0
            orientation_standing = 0.0
            base_height = -5.0
            torques_walking = 0.0
            torques_standing = 0.0
            energy_square_walking = 0.0
            energy_square_standing = 0.0
            base_height_walking = 0.0
            base_height_standing = 0.0
            penalty_lin_vel_y = 0.#-10.
        base_height_target = 0.55
        class arm_scales:
            arm_termination = None
            tracking_ee_sphere = 0.
            tracking_ee_world = 0.8
            tracking_ee_sphere_walking = 0.0
            tracking_ee_sphere_standing = 0.0
            tracking_ee_cart = None
            arm_orientation = None
            arm_energy_abs_sum = None
            tracking_ee_orn = 0.
            tracking_ee_orn_ry = None

class B1Z1RoughCfgPPO(ManipLocoCfgPPO):
    class policy (ManipLocoCfgPPO.policy):
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains

    class algorithm (ManipLocoCfgPPO.algorithm):
        torque_supervision = B1Z1RoughCfg.control.torque_supervision  #alert: also appears above
        adaptive_arm_gains = B1Z1RoughCfg.control.adaptive_arm_gains

    class runner (ManipLocoCfgPPO.runner):
        max_iterations = 40000
        experiment_name = 'b1z1_v2'
