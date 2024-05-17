# This file is used to save the demonstration data. It is borrowed from https://github.com/ZiwenZhuang/parkour/tree/main.

import os
import os.path as osp
import json
import pickle

import numpy as np
import torch

from skrl.memories.torch.random import RandomMemory


class DemonstrationSaver:
    def __init__(self, env, policy, policy_state_preprocessor,save_dir, rollout_storage_length=64, min_timesteps=1e6,
                 min_episodes=10000, success_traj_only=False, use_critic_obs=False, obs_disassemble_mapping=None):
        """
        Args:
            obs_disassemble_mapping (dict, optional): If set, the obs segment will be compressed using given type.
            example: {"forward_depth": "normalized_image", "forward_rgb": "normalized_image"}
            Defaults to None.
        """
        self.env = env
        self.policy = policy
        
        self.save_dir = save_dir
        self.rollout_storage_length = rollout_storage_length
        self.min_timesteps = min_timesteps
        self.min_episodes = min_episodes
        self.use_critic_obs = use_critic_obs
        self.success_traj_only = success_traj_only
        self.obs_disassemble_mapping = obs_disassemble_mapping
        self.policy_state_preprocessor = policy_state_preprocessor
        
        # self.rollout_storage = [] TODO: add this later; using skrl memory storage;
        
    def init_traj_handlers(self):
        if len(os.listdir(self.save_dir)) > 1:
            print("Continuing from previous data. You have to make sure the environment configuration is the same.")
            prev_traj = [x for x in os.listdir(self.save_dir) if x.startswith("trajectory")]
            prev_traj.sort(key=lambda x: int(x.split("_")[1]))
            # fill up the traj_idxs
            self.traj_idxs = []
            for traj in prev_traj:
                if len(os.listdir(osp.join(self.save_dir, traj))) == 0:
                    self.traj_idxs.append(int(traj.split("_")[1]))
            if len(self.traj_idxs) < self.env.num_envs:
                max_traj_idx = max(self.traj_idxs) if len(self.traj_idxs) > 0 else int(prev_traj[-1].split("_")[1])
                for _ in range(self.env.num_envs - len(self.traj_idxs)):
                    self.traj_idxs.append(max_traj_idx + 1)
                    max_traj_idx += 1
            self.traj_idxs = np.array(self.traj_idxs[:self.env.num_envs])
            with open(osp.join(self.save_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
            self.total_traj_completed = metadata["total_trajectories"]
            self.total_timesteps = metadata["total_timesteps"]
        else:
            self.traj_idxs = np.arange(self.env.num_envs)
            self.total_traj_completed = 0
            self.total_timesteps = 0
        self.metadata["total_timesteps"] = self.total_timesteps
        self.metadata["total_trajectories"] = self.total_traj_completed
        for traj_idx in self.traj_idxs:
            os.makedirs(osp.join(self.save_dir, "trajectory_{}".format(traj_idx)), exist_ok=True)
        self.dumped_traj_lengths = np.zeros(self.env.num_envs, dtype=np.int32)
        
        if self.obs_disassemble_mapping is not None:
            self.metadata["obs_segments"] = self.env.obs_segments
            self.metadata["obs_disassemble_mapping"] = self.obs_disassemble_mapping
            
    def init_storage_buffer(self):
        # TODO: check this later
        self.rollout_storage = RandomMemory(memory_size=self.rollout_storage_length, num_envs=self.env.num_envs, device=self.env.device)
        self.rollout_storage.create_tensor(name="obs", size=self.env.num_obs, dtype=torch.float)
        self.rollout_storage.create_tensor(name="action", size=self.env.num_acts, dtype=torch.float)
        self.rollout_storage.create_tensor(name="rewards", size=1, dtype=torch.float)
        self.rollout_storage.create_tensor(name="terminated", size=1, dtype=torch.bool)
        self.rollout_storage.create_tensor(name="log_prob", size=1, dtype=torch.float)
        self.rollout_storage.create_tensor(name="values", size=1, dtype=torch.float)
        self.rollout_storage.create_tensor(name="returns", size=1, dtype=torch.float)
        self.rollout_storage.create_tensor(name="advantages", size=1, dtype=torch.float)
        
        self.transition_has_timouts = False
        self.transition_timeouts = torch.zeros(self.rollout_storage_length, self.env.num_envs, dtype=torch.bool, device=self.env.device)
        
    def check_stop(self):
        return (self.total_traj_completed >= self.min_episodes) \
            and (self.total_timesteps >= self.min_timesteps)
            
    @torch.no_grad()
    def collect_step(self, step_i):
        """ Collect one step of demonstration data.
        """
        actions, rewards, dones, infos, n_obs = self.get_transition()
        self.add_transition(step_i, actions, rewards, dones, infos)
        # TODO: modify this later
        n_obs, _ = self.env.reset()
        self.obs = n_obs
    
    def get_transition(self):
        if self.use_critic_obs:
            raise NotImplementedError
        else:
            actions, log_prob, _ = self.policy.act({"states": self.policy_state_preprocessor(self.obs)}, role="policy")
            self.log_prob = log_prob
        n_obs, rewards, dones, _, infos = self.env.step(actions)
        return actions, rewards, dones, infos, n_obs
    
    def add_transition(self, step_i, actions, rewards, dones, info):
        self.rollout_storage.add_samples(states=self.obs, actions=actions, rewards=rewards, terminated=dones, log_prob=self.log_prob)
        if "time_outs" in info:
            self.transition_has_timouts = True
            self.transition_timeouts[step_i] = info["time_outs"]
    
    def dump_to_file(self, env_i, step_slice):
        """ Dump the part of trajectory to the trajectory directory.
        """
        traj_idx = self.traj_idxs[env_i]
        traj_dir = osp.join(self.save_dir, "trajectory_{}".format(traj_idx))
        traj_file = osp.join(
            traj_dir,
            f"traj_{self.dumped_traj_lengths[env_i]:06d}_{self.dumped_traj_lengths[env_i]+step_slice.stop-step_slice.start:06d}.pickle",
        )
        trajectory = self.wrap_up_trajectory(env_i, step_slice)
        with open(traj_file, "wb") as f:
            pickle.dump(trajectory, f)
        self.dumped_traj_lengths[env_i] += step_slice.stop - step_slice.start
        self.total_timesteps += step_slice.stop - step_slice.start
        with open(osp.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
    
    def wrap_up_trajectory(self, env_i, step_slice):
        trajectory = dict(
            actions=self.rollout_storage.get_tensor_by_name("action")[step_slice, env_i].cpu().numpy(),
            rewards=self.rollout_storage.get_tensor_by_name("rewards")[step_slice, env_i].cpu().numpy(),
            dones=self.rollout_storage.get_tensor_by_name("terminated")[step_slice, env_i].cpu().numpy(),
            values=self.rollout_storage.get_tensor_by_name("values")[step_slice, env_i].cpu().numpy(),
        )
        # compress observations components if set
        if self.obs_disassemble_mapping is not None:
            observations = self.rollout_storage.get_tensor_by_name("obs")[step_slice, env_i].cpu().numpy()
            for component_name in self.meta_data["obs_segments"].keys():
                # TODO: add this later
                raise NotImplementedError
        else:
            trajectory["observations"] = self.rollout_storage.get_tensor_by_name("obs")[step_slice, env_i].cpu().numpy()
        if self.transition_has_timouts:
            trajectory["timeouts"] = self.transition_timeouts[step_slice, env_i].cpu().numpy()
        return trajectory
    
    def update_traj_handler(self, env_i, step_slice):
        traj_idx = self.traj_idxs[env_i]
        
        if self.success_traj_only:
            if self.rollout_storage.get_tensor_by_name("terminated")[step_slice.stop-1, env_i] and \
                not self.transition_timeouts[step_slice.stop-1, env_i]:
                # done by termination not timeout (failed)
                # remove all files in the trajectory directory
                traj_dir = osp.join(self.save_dir, f"trajectory_{traj_idx}")
                for traj in os.listdir(traj_dir):
                    try:
                        if traj.startswith("traj_"):
                            start_timestep, stop_timestep = traj.split("_")[1:]
                            start_timestep = int(start_timestep)
                            stop_timestep = int(stop_timestep)
                            self.total_timesteps -= stop_timestep - start_timestep
                    except:
                        pass
                    os.remove(osp.join(traj_dir, traj))
                self.dumped_traj_lengths[env_i] = 0
                return
        # update the handlers to a new trajectory
        # Also, skip the trajectory directory that has data collected before this run.
        while len(os.listdir(osp.join(self.save_dir, f"trajectory_{traj_idx}"))) > 0:
            traj_idx = max(self.traj_idxs) + 1
            os.makedirs(osp.join(self.save_dir, f"trajectory_{traj_idx}"), exist_ok=True)
        self.traj_idxs[env_i] = traj_idx
        self.total_traj_completed += 1
        self.dumped_traj_lengths[env_i] = 0
    
    def save_steps(self):
        """ dump a series or transitions to the file."""
        for rollout_env_i in range(self.env.num_envs):
            done_idxs = torch.where(self.rollout_storage.get_tensor_by_name("terminated")[:, rollout_env_i, 0])[0]
            if len(done_idxs) == 0:
                # dump the whole rollout for this env
                self.dump_to_file(rollout_env_i, slice(0, self.rollout_storage_length))
            else:
                start_idx = 0
                di = 0
                while di < done_idxs.shape[0]:
                    end_idx = done_idxs[di].item()
                    
                    # dump and update the traj_idx for this env
                    self.dump_to_file(rollout_env_i, slice(start_idx, end_idx+1))
                    self.update_traj_handler(rollout_env_i, slice(start_idx, end_idx+1))
                    
                    start_idx = end_idx + 1
                    di += 1
    
    def collect_and_save(self, config=None):
        """
        Run the rollout to collect the demonstration data and save it to file.
        """
        # create directory and save metadata file
        self.metadata = {
            "config": config,
            "env": self.env.__class__.__name__,
            "policy": self.policy.__class__.__name__,
            "rollout_storage_length": self.rollout_storage_length,
            "success_traj_only": self.success_traj_only,
            "min_timesteps": self.min_timesteps,
            "min_episodes": self.min_episodes,
        }
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.init_traj_handlers()
        self.init_storage_buffer()
        
        with open(osp.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
            
        # collect the demonstration data
        self.env.reset()
        obs = self.env.get_observations() # TODO: modify this later, may not be correct
        self.obs = obs
        while not self.check_stop():
            for step_i in range(self.rollout_storage_length):
                self.collect_step(step_i)
            self.save_steps()
            self.rollout_storage.reset()
            self.print_log()
        
    def print_log(self):
        print("total timesteps: {}".format(self.total_timesteps))
        print("total trajectories: {}".format(self.total_traj_completed))
        
    def __del__(self):
        """ In case the process stops unexpectedly, close the file handlers."""
        for traj_idx in self.traj_idxs:
            traj_dir = osp.join(self.save_dir, "trajectory_{}".format(traj_idx))
            # remove empty directories
            if len(os.listdir(traj_dir)) == 0:
                os.rmdir(traj_dir)
        for timestep_count in self.dumped_traj_lengths:
            self.total_timesteps += timestep_count
        self.metadata["total_timesteps"] = self.total_timesteps.item() if isinstance(self.total_timesteps, np.int64) else self.total_timesteps
        self.metadata["total_trajectories"] = self.total_traj_completed
        with open(osp.join(self.save_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=4)
        print(f"Saved dataset in {self.save_dir}")
        