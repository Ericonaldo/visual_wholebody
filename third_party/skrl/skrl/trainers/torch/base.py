from typing import List, Optional, Union

import atexit
import tqdm

import torch

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper


def generate_equally_spaced_scopes(num_envs: int, num_simultaneous_agents: int) -> List[int]:
    """Generate a list of equally spaced scopes for the agents

    :param num_envs: Number of environments
    :type num_envs: int
    :param num_simultaneous_agents: Number of simultaneous agents
    :type num_simultaneous_agents: int

    :raises ValueError: If the number of simultaneous agents is greater than the number of environments

    :return: List of equally spaced scopes
    :rtype: List[int]
    """
    scopes = [int(num_envs / num_simultaneous_agents)] * num_simultaneous_agents
    if sum(scopes):
        scopes[-1] += num_envs - sum(scopes)
    else:
        raise ValueError(f"The number of simultaneous agents ({num_simultaneous_agents}) is greater than the number of environments ({num_envs})")
    return scopes


class Trainer:
    def __init__(self,
                 env: Wrapper,
                 agents: Union[Agent, List[Agent]],
                 agents_scope: Optional[List[int]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class for trainers

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``)
        :type cfg: dict, optional
        """
        self.cfg = cfg if cfg is not None else {}
        self.env = env
        self.agents = agents
        self.agents_scope = agents_scope if agents_scope is not None else []

        # get configuration
        self.timesteps = self.cfg.get("timesteps", 0)
        self.headless = self.cfg.get("headless", False)
        self.disable_progressbar = self.cfg.get("disable_progressbar", False)
        self.close_environment_at_exit = self.cfg.get("close_environment_at_exit", True)

        self.initial_timestep = self.cfg.get("initial_timestep", 0)

        # setup agents
        self.num_simultaneous_agents = 0
        self._setup_agents()

        # register environment closing if configured
        if self.close_environment_at_exit:
            @atexit.register
            def close_env():
                logger.info("Closing environment")
                self.env.close()
                logger.info("Environment closed")
            
        self.record_video = cfg.get("record_video", False)

    def __str__(self) -> str:
        """Generate a string representation of the trainer

        :return: Representation of the trainer as string
        :rtype: str
        """
        string = f"Trainer: {self}"
        string += f"\n  |-- Number of parallelizable environments: {self.env.num_envs}"
        string += f"\n  |-- Number of simultaneous agents: {self.num_simultaneous_agents}"
        string += "\n  |-- Agents and scopes:"
        if self.num_simultaneous_agents > 1:
            for agent, scope in zip(self.agents, self.agents_scope):
                string += f"\n  |     |-- agent: {type(agent)}"
                string += f"\n  |     |     |-- scope: {scope[1] - scope[0]} environments ({scope[0]}:{scope[1]})"
        else:
            string += f"\n  |     |-- agent: {type(self.agents)}"
            string += f"\n  |     |     |-- scope: {self.env.num_envs} environment(s)"
        return string

    def _setup_agents(self) -> None:
        """Setup agents for training

        :raises ValueError: Invalid setup
        """
        # validate agents and their scopes
        if type(self.agents) in [tuple, list]:
            # single agent
            if len(self.agents) == 1:
                self.num_simultaneous_agents = 1
                self.agents = self.agents[0]
                self.agents_scope = [1]
            # parallel agents
            elif len(self.agents) > 1:
                self.num_simultaneous_agents = len(self.agents)
                # check scopes
                if not len(self.agents_scope):
                    logger.warning("The agents' scopes are empty, they will be generated as equal as possible")
                    self.agents_scope = [int(self.env.num_envs / len(self.agents))] * len(self.agents)
                    if sum(self.agents_scope):
                        self.agents_scope[-1] += self.env.num_envs - sum(self.agents_scope)
                    else:
                        raise ValueError(f"The number of agents ({len(self.agents)}) is greater than the number of parallelizable environments ({self.env.num_envs})")
                elif len(self.agents_scope) != len(self.agents):
                    raise ValueError(f"The number of agents ({len(self.agents)}) doesn't match the number of scopes ({len(self.agents_scope)})")
                elif sum(self.agents_scope) != self.env.num_envs:
                    raise ValueError(f"The scopes ({sum(self.agents_scope)}) don't cover the number of parallelizable environments ({self.env.num_envs})")
                # generate agents' scopes
                index = 0
                for i in range(len(self.agents_scope)):
                    index += self.agents_scope[i]
                    self.agents_scope[i] = (index - self.agents_scope[i], index)
            else:
                raise ValueError("A list of agents is expected")
        else:
            self.num_simultaneous_agents = 1

    def train(self) -> None:
        """Train the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def eval(self) -> None:
        """Evaluate the agents

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

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

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
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

        # reset env
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
            traj_length = 50000 # 35000 # min(int(self.env.max_episode_length), 300)
        else:
            traj_length = min(int(self.env.max_episode_length), 150)
        
        for timestep in tqdm.tqdm(range(0, traj_length), disable=self.disable_progressbar):

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

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

                # write data to TensorBoard
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states
                
        if self.record_video:
            for mp4_writer in mp4_writers:
                mp4_writer.close()

    def multi_agent_train(self) -> None:
        """Train multi-agents

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
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = infos.get("shared_states", None)

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # pre-interaction
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = infos.get("shared_states", None)
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)

            # post-interaction
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            with torch.no_grad():
                if not self.env.agents:
                    states, infos = self.env.reset()
                    shared_states = infos.get("shared_states", None)
                else:
                    states = next_states
                    shared_states = shared_next_states

    def multi_agent_eval(self) -> None:
        """Evaluate multi-agents

        This method executes the following steps in loop:

        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Reset environments
        """
        assert self.num_simultaneous_agents == 1, "This method is not allowed for simultaneous agents"
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = infos.get("shared_states", None)

        for timestep in tqdm.tqdm(range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar):

            # compute actions
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = infos.get("shared_states", None)
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                self.agents.record_transition(states=states,
                                              actions=actions,
                                              rewards=rewards,
                                              next_states=next_states,
                                              terminated=terminated,
                                              truncated=truncated,
                                              infos=infos,
                                              timestep=timestep,
                                              timesteps=self.timesteps)
                super(type(self.agents), self.agents).post_interaction(timestep=timestep, timesteps=self.timesteps)

                # reset environments
                if not self.env.agents:
                    states, infos = self.env.reset()
                    shared_states = infos.get("shared_states", None)
                else:
                    states = next_states
                    shared_states = shared_next_states
