from typing import List, Optional, Union

import copy
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.trainers.torch import Trainer
from skrl.envs.wrappers.torch import Wrapper

DAGGER_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,            # number of timesteps to train for
    "headless": False,              # whether to use headless mode (no rendering)
    "disable_progressbar": False,   # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,   # whether to close the environment on normal program termination
    "teacher_pretrain": False,      # whether to use teacher actions to step first
}

class DAggerTrainer(Trainer):
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 teacher_agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        _cfg = copy.deepcopy(DAGGER_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        self.teacher_agents = teacher_agents
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)
        self.teacher_pretrain = self.cfg.get("teacher_pretrain", False)
        if self.teacher_pretrain:
            self.pretrain_percentage = 0.2
        else:
            self.pretrain_percentage = 0.0
        
        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)
            
    def train(self) -> None:
        """Train the agents with DAgger algorithm

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """
        # set running mode
        self.teacher_agents.set_running_mode("eval")
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")
            
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_train()
            else:
                raise NotImplementedError("DAgger is not implemented for multi-agent environments")
        else:
            # multi-agent
            raise NotImplementedError("DAgger is not implemented for multi-agent environments")
        
    def eval(self) -> None:
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")
            
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                self.single_agent_eval()
            else:
                raise NotImplementedError("DAgger is not implemented for multi-agent environments")
        else:
            # multi-agent
            raise NotImplementedError("DAgger is not implemented for multi-agent environments")
    
    def single_agent_train(self) -> None:
        """Train agent

        This method executes the following steps in loop:

        - Pre-interaction
        - Compute actions
        - Interact with the environments
        - Render scene
        - Record transitions
        - Post-interaction
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"
        
        # reset env
        states, infos = self.env.reset()
        
        # threshold_timestep = int(self.pretrain_percentage * self.timesteps)
        threshold_timestep = self.cfg.get("pretrain_timesteps", 4000)
        
        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):
            student_obs = states["states"]
            teacher_obs = states["obs"]
            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            self.teacher_agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            
            with torch.no_grad():
                teacher_actions = self.teacher_agents.act(teacher_obs, timestep=timestep, timesteps=self.timesteps)[0]
                actions = self.agents.act(student_obs, timestep=timestep, timesteps=self.timesteps)[0]
                
                if timestep < threshold_timestep:
                    next_states, rewards, terminated, truncated, infos = self.env.step(teacher_actions)
                else:
                    next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                
                if infos.get("replaced_action", None) is not None: # Use replaced action to learn
                    teacher_actions = infos.get("replaced_action")
                    
                # render scene
                if not self.headless:
                    self.env.render()
                
                # record the environments' transitions
                self.agents.record_transition(student_obs=student_obs,
                                              teacher_obs=teacher_obs, # teacher observations
                                              actions=actions,
                                              teacher_actions=teacher_actions, # teacher actions label
                                              rewards=rewards,
                                              next_states=next_states["obs"],
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)
                
            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            
            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
                
    def single_agent_eval(self) -> None:
        """Evaluate agent

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents == 1, "This method is not allowed for multi-agents"

        states, infos = self.env.reset()
        
        mp4_writers = []
        if self.record_video:
            import imageio, os
            self.env.enable_viewer_sync = False
            for i in range(self.env.num_envs):
                video_name = f"{i}.mp4"
                run_dir = self.cfg["log_dir"]
                path = f"../logs/videos/{run_dir}/{self.cfg['video_name']}"
                if not os.path.exists(path):
                    os.makedirs(path)
                video_name = os.path.join(path, video_name)
                mp4_writer = imageio.get_writer(video_name, fps=10)
                mp4_writers.append(mp4_writer)

        if not self.record_video:
            traj_length = 14000 # int(self.env.max_episode_length)
        else:
            traj_length = int(self.env.max_episode_length)
        
        for timestep in tqdm.tqdm(range(0, traj_length), disable=self.disable_progressbar):

            # if timestep > 20:
            #     break
            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states["states"], timestep=timestep, timesteps=self.timesteps)[0]
                teacher_actions = self.teacher_agents.act(states["obs"], timestep=timestep, timesteps=self.timesteps)[0]
                # print("actions: ", actions[1])
                # action_hist.append(actions[1].cpu().numpy())

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()
                    
                if self.record_video:
                    imgs = self.env.render_record(mode='rgb_array')
                    if imgs is not None:
                        for i in range(self.env.num_envs):
                            mp4_writers[i].append_data(imgs[i])
                
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    # action_hist = np.array(action_hist)
                    # np.save("action_hist_fix_1.npy", action_hist)
                    states, infos = self.env.reset()
            else:
                states = next_states
                
        if self.record_video:
            for mp4_writer in mp4_writers:
                mp4_writer.close()
    
