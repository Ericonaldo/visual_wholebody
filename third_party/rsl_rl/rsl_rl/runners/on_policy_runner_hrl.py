import time
import os
from collections import deque
import statistics

import torch

from rsl_rl.algorithms import PPOHRL, PPO
from rsl_rl.modules import ActorCriticHRL, ActorCriticRecurrent, ActorCritic
from rsl_rl.env import VecEnv

import wandb
from torchinfo import summary

class OnPolicyRunnerHRL:
    def __init__(self, 
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):
        
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.low_policy_cfg = train_cfg["low_level_policy"]["policy"]
        self.pretrained_low_level_policy_path = self.cfg["pretrained_low_level_policy_path"]
        self.device = device
        self.env = env
        
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCriticHRL
        actor_critic: ActorCriticHRL = actor_critic_class(self.env.num_task_obs,
                                                          self.env.num_task_obs,
                                                          self.env.num_task_actions,
                                                          **self.policy_cfg).to(self.device)
        
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPOHRL
        self.alg: PPOHRL = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        summary(self.alg.actor_critic)
        
        # low level policy
        self.low_level_policy = self._load_low_level_policy()
        
        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_task_obs],
                              [self.env.num_task_obs], [self.env.num_task_actions])
        
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
        _, _ = self.env.reset()
    
    def _load_low_level_policy(self, stochastic=False):
        low_actor_critic_class = eval(self.cfg["low_level_policy_class_name"]) # ActorCritic
        low_actor_critic: ActorCritic = low_actor_critic_class(self.env.cfg.env.num_proprio,
                                                               self.env.cfg.env.num_proprio,
                                                               self.env.num_actions,
                                                               **self.low_policy_cfg,
                                                               num_priv=self.env.cfg.env.num_priv,
                                                               num_hist=self.env.cfg.env.history_len,
                                                               num_prop=self.env.cfg.env.num_proprio,)
        loaded_dict = torch.load(self.pretrained_low_level_policy_path, map_location=self.device)
        low_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        low_actor_critic = low_actor_critic.to(self.device)
        
        print("Low level pretrained policy loaded!")
        if not stochastic:
            return low_actor_critic.act_inference
        else:
            return low_actor_critic.act
        
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        task_obs = self.env.get_task_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, task_obs = obs.to(self.device), critic_obs.to(self.device), task_obs.to(self.device)
        
        self.alg.actor_critic.train()
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        donebuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            inst_reward_max = -100. * torch.ones(self.env.num_envs, dtype=torch.float, device=self.device)
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    command_actions = self.alg.act(task_obs.detach(), task_obs.detach())
                    self.env.convert_high_level_to_command(command_actions) # Take command
                    obs = self.env.get_observations()
                    low_level_actions = self.low_level_policy(obs.detach(), hist_encoding=True)
                    obs, privileged_obs, task_obs, rewards, dones, infos = self.env.step(low_level_actions.detach())
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, task_obs = obs.to(self.device), critic_obs.to(self.device), task_obs.to(self.device)
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    inst_reward_max = torch.maximum(inst_reward_max, rewards)
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        donebuffer.append(len(new_ids) / self.env.num_envs)
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    if dones.any():
                        obs, task_obs = self.env.reset()
                
                stop = time.time()
                collection_time = stop - start
                
                start = stop
                self.alg.compute_returns(task_obs)
                
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
        
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        
        ep_string = f''
        wandb_dict = {}
        
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                
                if "rew" in key:
                    wandb_dict["Episode_rew/" + key] = value
                elif "metric" in key:
                     wandb_dict['Episode_metric/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
                
        action_std = self.alg.actor_critic.std.mean()
        std_numpy = self.alg.actor_critic.std.detach().cpu().numpy()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Policy/mean_noise_std'] = action_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection_time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            wandb_dict['Train/dones'] = statistics.mean(locs['donebuffer'])
            wandb_dict['Train/inst_max_reward'] = max(locs['inst_reward_max'])
        
        wandb.log(wandb_dict, step=locs['it'])
        
        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {action_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Dones:':>{pad}} {statistics.mean(locs['donebuffer']):.2f}\n""")
            
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {action_std.item():.2f}\n"""
                          f"""{'action noise std distribution:':>{pad}} {std_numpy.tolist()}\n""")
            
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)
        
    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos
        }, path)
        
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None, stochastic=False):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        if not stochastic:
            return self.alg.actor_critic.act_inference
        else:
            return self.alg.actor_critic.act
