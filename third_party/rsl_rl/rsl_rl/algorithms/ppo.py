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

from time import time
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 mixing_schedule=[0.5, 2000, 4000], 
                 torque_supervision=True,
                 torque_supervision_schedule=[0.1, 1000, 1000],
                 adaptive_arm_gains=True,
                 min_policy_std=None,
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.actor.history_encoder.parameters(), lr=learning_rate)
        self.priv_reg_coef_schedual = priv_reg_coef_schedual

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.min_policy_std = torch.tensor(min_policy_std, device=self.device)

        self.mixing_schedule = mixing_schedule
        self.torque_supervision = torque_supervision
        self.torque_supervision_schedule = torque_supervision_schedule
        self.adaptive_arm_gains = adaptive_arm_gains
        self.counter = 0

        # adaptive arm gains
        if self.adaptive_arm_gains:
            self.arm_fk = self.arm_fk_adaptive_gains
        else:
            self.arm_fk = self.arm_fk_fixed_gains

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, hist_encoding=False):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, arm_rewards, dones, infos):
        self.transition.rewards = torch.stack([rewards.clone(), arm_rewards.clone()], dim=-1)
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        
        if 'target_arm_torques' in infos:
            self.transition.target_arm_torques = infos['target_arm_torques'].detach()
            self.transition.current_arm_dof_pos = infos['current_arm_dof_pos'].detach()
            self.transition.current_arm_dof_vel = infos['current_arm_dof_vel'].detach()

            # Record the transition
            self.storage.add_transitions(self.transition, torque_supervision=True)
        else:
            self.storage.add_transitions(self.transition, torque_supervision=False)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_arm_torques_loss = 0
        mean_priv_reg_loss = 0
        value_mixing_ratio = self.get_value_mixing_ratio()
        torque_supervision_weight = self.get_torque_supervision_weight() if self.torque_supervision else 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, target_arm_torques, current_arm_dof_pos, current_arm_dof_vel, hid_states_batch, masks_batch in generator:

                self.actor_critic.act(obs_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # Adaptation module update
                priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                with torch.inference_mode():
                    hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]
                # priv_reg_loss = torch.zeros(1, device=self.device)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                only_train_leg = False

                mixing_advantages_batch = torch.zeros_like(advantages_batch)
                if only_train_leg == True:
                    mixing_advantages_batch[..., 0] = advantages_batch[..., 0]
                    mixing_advantages_batch[..., 1] = advantages_batch[..., 1]
                else:
                    mixing_advantages_batch[..., 0] = advantages_batch[..., 0] + value_mixing_ratio * advantages_batch[..., 1]
                    mixing_advantages_batch[..., 1] = advantages_batch[..., 1] + value_mixing_ratio * advantages_batch[..., 0]
                ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
                surrogate = - mixing_advantages_batch * ratio
                surrogate_clipped = - mixing_advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                if only_train_leg == True:
                    surrogate_loss = torch.max(surrogate, surrogate_clipped)[:, 0].mean()
                else:
                    surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss \
                       + self.value_loss_coef * value_loss \
                       - self.entropy_coef * entropy_batch.mean() \
                       + priv_reg_coef * priv_reg_loss


                # adaptive arm gains
                if self.adaptive_arm_gains:
                    actions_mean = self.actor_critic.act_inference(obs_batch)
                    target_arm_dof_pos = actions_mean[:, 12: -6]
                    delta_arm_p_gains = actions_mean[:, -6:]
                else:
                    target_arm_dof_pos = self.actor_critic.act_inference(obs_batch)[:, -6:]
                    delta_arm_p_gains = None

                # arm torque supervision
                if self.torque_supervision:
                    arm_torques = self.arm_fk(delta_arm_p_gains, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel)
                    arm_torques_loss = (arm_torques - target_arm_torques).pow(2).mean()
                    torque_supervision_weight = self.get_torque_supervision_weight()
                    loss += arm_torques_loss * torque_supervision_weight
                    mean_arm_torques_loss += arm_torques_loss.item()
                

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_priv_reg_loss += priv_reg_loss.item()
                
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_arm_torques_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        self.storage.clear()

        self.update_counter()

        self.enforce_min_std()

        return mean_value_loss, mean_surrogate_loss, mean_arm_torques_loss, value_mixing_ratio, torque_supervision_weight, mean_priv_reg_loss, priv_reg_coef
    
    def update_dagger(self):
        mean_hist_latent_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, target_arm_torques, current_arm_dof_pos, current_arm_dof_vel, hid_states_batch, masks_batch in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0])

                # Adaptation module update
                with torch.inference_mode():
                    priv_latent_batch = self.actor_critic.actor.infer_priv_latent(obs_batch)
                hist_latent_batch = self.actor_critic.actor.infer_hist_latent(obs_batch)
                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.actor.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss

    def enforce_min_std(self):
        current_std = self.actor_critic.std.detach()
        new_std = torch.max(current_std, self.min_policy_std).detach()
        self.actor_critic.std.data = new_std
    
    def update_counter(self):
        self.counter += 1
    
    def get_value_mixing_ratio(self):
        return min(max((self.counter - self.mixing_schedule[1]) / self.mixing_schedule[2], 0), 1) * self.mixing_schedule[0]
    
    def get_torque_supervision_weight(self):
        return (1 - min(max((self.counter - self.torque_supervision_schedule[1]) / self.torque_supervision_schedule[2], 0), 1)) * self.torque_supervision_schedule[0]

    def set_arm_default_coeffs(self, default_arm_p_gains, default_arm_d_gains, default_arm_dof_pos):
        self.default_arm_p_gains = default_arm_p_gains
        self.default_arm_d_gains = default_arm_d_gains
        self.default_arm_dof_pos = default_arm_dof_pos
    
    def arm_fk_adaptive_gains(self, delta_arm_p_gains, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel):
        adaptive_arm_p_gains = self.default_arm_p_gains + delta_arm_p_gains
        adaptive_arm_d_gains = 2 * (adaptive_arm_p_gains ** 0.5)
        arm_torques = adaptive_arm_p_gains * (target_arm_dof_pos + self.default_arm_dof_pos - current_arm_dof_pos) \
            - adaptive_arm_d_gains * current_arm_dof_vel
        return arm_torques

    def arm_fk_fixed_gains(self, _, target_arm_dof_pos, current_arm_dof_pos, current_arm_dof_vel):
        fixed_arm_p_gains = self.default_arm_p_gains
        fixed_arm_d_gains = self.default_arm_d_gains
        arm_torques = fixed_arm_p_gains * (target_arm_dof_pos + self.default_arm_dof_pos - current_arm_dof_pos) \
            - fixed_arm_d_gains * current_arm_dof_vel
        return arm_torques

