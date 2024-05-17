from typing import Any, Dict, Optional, Tuple, Union

import copy
import itertools
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR

# [start-config-dict-torch]
DAGGER_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]


class DAgger_RNN(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 state_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        
        """
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        
        _cfg = copy.deepcopy(DAGGER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models, memory=memory, observation_space=observation_space,
                         action_space=action_space, device=device, cfg=_cfg)
        
        self.state_space = state_space
        
        # models
        self.policy = self.models.get("policy", None)
        
        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        
        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]

        self._kl_threshold = self.cfg["kl_threshold"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]
        
        self._fixed_base = self.cfg["fixed_base"]
        self._reach_only = self.cfg["reach_only"]
        
        if self.policy is not None:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])
            self.checkpoint_modules["optimizer"] = self.optimizer
        
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
            
    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")
        
        if self.memory is not None:
            # self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="student_obs", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="teacher_obs", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="teacher_actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            
            # tensors sampled during training
            self._tensor_names = ["student_obs", "teacher_obs", "actions", "teacher_actions", "rewards", "terminated", "log_prob"]
        
        # RNN specifications
        self._rnn = False # flag to indicate whether RNN is available
        self._rnn_tensors_names = [] # used for sampling during training
        self._rnn_final_states = {"policy": []}
        self._rnn_initial_states = {"policy": []}
        self._rnn_sequence_length = self.policy.get_specification().get("rnn", {}).get("sequence_length", 1)
        
        # policy
        for i, size in enumerate(self.policy.get_specification().get("rnn", {}).get("sizes", [])):
            self._rnn = True
            # create tensors in memory
            if self.memory is not None:
                self.memory.create_tensor(name=f"rnn_policy_{i}", size=(size[0], size[2]), dtype=torch.float32, keep_dimensions=True)
                self._rnn_tensors_names.append(f"rnn_policy_{i}")
            # default RNN states
            self._rnn_initial_states["policy"].append(torch.zeros(size, dtype=torch.float32, device=self.device))
            
        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None
        
    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        rnn = {"rnn": self._rnn_initial_states["policy"]} if self._rnn else {}
        
        # sample random actions
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states, **rnn}, role="policy")
        
        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": states, **rnn}, role="policy")
        if log_prob is None:
            log_prob = 0
        self._current_log_prob = log_prob
        
        if self._rnn:
            self._rnn_final_states["policy"] = outputs.get("rnn", [])
            self._rnn_initial_states["policy"] = self._rnn_final_states["policy"]
        
        return actions, log_prob, outputs
    
    def record_transition(self, student_obs: torch.Tensor, teacher_obs, actions: torch.Tensor, teacher_actions: torch.Tensor, rewards: torch.Tensor, next_states: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: Any, timestep: int, timesteps: int) -> None:
        super().record_transition(states=teacher_obs, actions=actions, rewards=rewards, next_states=next_states,
                                  terminated=terminated, truncated=truncated, infos=infos, timestep=timestep,
                                  timesteps=timesteps)
        
        if self.memory is not None:
            self._current_next_states = next_states
            
            # package RNN states
            rnn_states = {}
            if self._rnn:
                rnn_states.update({f"rnn_policy_{i}": s.transpose(0, 1) for i, s in enumerate(self._rnn_final_states["policy"])})
            
            self.memory.add_samples(student_obs=student_obs, teacher_obs=teacher_obs, actions=actions, teacher_actions=teacher_actions, rewards=rewards,
                                    terminated=terminated, log_prob=self._current_log_prob, **rnn_states)
        # update RNN states
        if self._rnn:
            # reset states if the episodes have ended
            finished_episodes = terminated.nonzero(as_tuple=False)
            if finished_episodes.numel():
                for rnn_state in self._rnn_final_states["policy"]:
                    rnn_state[:, finished_episodes[:, 0]] = 0
            
            self._rnn_initial_states = self._rnn_final_states
            
    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        pass
    
    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")
        
        # wirte tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)
        
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        
        rnn_policy = {}
        
            
        cumulative_dagger_loss = 0
        cumulative_entropy_loss = 0
        
        # learning epochs
        for epoch in range(self._learning_epochs):
            # compute returns and advantages
            sampled_batches = self.memory.sample_all(names=self._tensor_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)
            
            if self._rnn:
                sampled_rnn_batches = self.memory.sample_all(names=self._rnn_tensors_names, mini_batches=self._mini_batches, sequence_length=self._rnn_sequence_length)
        
            # kl_divergences = []
            
            # mini-batches loop
            for i, (sampled_student_obs, sampled_teacher_obs, sampled_actions, sampled_teacher_actions, sampled_rewards, sampled_terminated, sampled_log_prob) in enumerate(sampled_batches):
                
                if self._rnn:
                    rnn_policy = {"rnn": [s.transpose(0, 1) for s, n in zip(sampled_rnn_batches[i], self._rnn_tensors_names) if "policy" in n], "terminated": sampled_terminated}

                # _, next_log_prob, _ = self.policy.act({"states": sampled_student_obs, "taken_actions": sampled_actions, **rnn_policy}, role="policy")
                
                # # compute approximate KL divergence
                # with torch.no_grad():
                #     ratio = next_log_prob - sampled_log_prob
                #     kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                #     kl_divergences.append(kl_divergence)
                    
                # early stopping with KL divergence
                # if self._kl_threshold and kl_divergence > self._kl_threshold:
                #     break
                
                 # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0
                
                # compute policy loss
                student_actions, _, _ = self.policy.act({"states": sampled_student_obs, **rnn_policy}, role="policy")
                # student_actions_gripper = student_actions[:, 6]
                # sampled_teacher_actions_gripper = sampled_teacher_actions[:, 6]
                # gripper_loss = F.binary_cross_entropy(student_actions_gripper, sampled_teacher_actions_gripper)
                # student_actions_other = torch.cat([student_actions[:, :6], student_actions[:, 7:]], dim=1)
                # sampled_teacher_actions_other = torch.cat([sampled_teacher_actions[:, :6], sampled_teacher_actions[:, 7:]], dim=1)
                # dagger_loss = F.mse_loss(student_actions_other, sampled_teacher_actions_other) + 0.01 * gripper_loss
                if self._fixed_base:
                    dagger_loss = F.mse_loss(student_actions[:, :-2], sampled_teacher_actions[:, :-2]) # TODO: only for fixed base robot
                elif self._reach_only:
                    dagger_loss = F.mse_loss(student_actions[:, :-3], sampled_teacher_actions[:, :-3]) + \
                        F.mse_loss(student_actions[:, -2:], sampled_teacher_actions[:, -2:])
                else:
                    dagger_loss = F.mse_loss(student_actions, sampled_teacher_actions)
                
                # optimization step
                self.optimizer.zero_grad()
                (dagger_loss + entropy_loss).backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                self.optimizer.step()
                
                # update cumulative losses
                cumulative_dagger_loss += dagger_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
            
            # # update learning rate
            # if self._learning_rate_scheduler:
            #     if isinstance(self.scheduler, KLAdaptiveLR):
            #         self.scheduler.step(torch.tensor(kl_divergences).mean())
            #     else:
            #         self.scheduler.step()
        
        # record data
        self.track_data("Loss / DAgger loss", cumulative_dagger_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
        
        if self._current_log_prob != 0:
            self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())
        
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        