from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
import torch
import os

import torch.nn as nn

from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from envs import *

from utils.config import load_cfg, get_params, copy_cfg
import utils.wrapper as wrapper

set_seed(43)

def create_env(cfg, args):
    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = args.small_value_set_zero
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)
    robot_start_pose = (-2.00, 0, 0.55)
    if args.eval:
        robot_start_pose = (-0.85, 0, 0.55)
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless, 
                         use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands, no_feature=args.no_feature, mask_arm=args.mask_arm, pitch_control=args.pitch_control,
                         rand_control=args.rand_control, arm_delay=args.arm_delay, robot_start_pose=robot_start_pose,
                         rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip, stop_pick=args.stop_pick, table_height=args.table_height, eval=args.eval)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, use_tanh=False, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", deterministic=False):
        Model.__init__(self, observation_space, action_space, device)
        transform_func = torch.distributions.transforms.TanhTransform() if use_tanh else None
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, transform_func=transform_func, deterministic=deterministic)

        self.num_features = num_features
        self.encode_dim = encode_dim
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512),
                                                  nn.ELU(),
                                                  nn.Linear(512, self.encode_dim),)
        self.net = nn.Sequential(nn.Linear(self.num_observations - self.num_features + self.encode_dim, 512),
                            nn.ELU(),
                            nn.Linear(512, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, self.num_actions)
                            )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        if self.num_features > 0:
            features_encode = self.feature_encoder(inputs["states"][..., :self.num_features])
            actions = self.net(torch.cat([inputs["states"][..., self.num_features:], features_encode], dim=-1))
        else:
            actions = self.net(inputs["states"])
        # actions[:, 6] = torch.sigmoid(actions[:, 6])
        return actions, self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        
        self.num_features = num_features
        self.encode_dim = encode_dim
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512),
                                                    nn.ELU(),
                                                    nn.Linear(512, self.encode_dim))

        self.net = nn.Sequential(nn.Linear(self.num_observations - self.num_features + self.encode_dim, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        if self.num_features > 0:
            feature_encode = self.feature_encoder(inputs["states"][..., :self.num_features])
            return self.net(torch.cat([inputs["states"][..., self.num_features:], feature_encode], dim=-1)), {}
        else:
            return self.net(inputs["states"]), {}

def get_trainer(is_eval=False):
    args = get_params()
    args.eval = is_eval
    args.wandb = args.wandb and (not args.eval) and (not args.debug)
    cfg_file = "b1z1_" + args.task[4:].lower() + ".yaml"
    file_path = "data/cfg/" + cfg_file
    
    if args.resume:
        experiment_dir = os.path.join(args.experiment_dir, args.wandb_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        pt_files = os.listdir(checkpoint_dir)
        pt_files = [file for file in pt_files if file.endswith(".pt") and (not file.startswith("best"))]
        # Find the latest checkpoint
        checkpoint_steps = 0
        if len(pt_files) > 0:
            args.checkpoint = os.path.join(checkpoint_dir, sorted(pt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1])
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        cfg_files = os.listdir(experiment_dir)
        cfg_files = [file for file in cfg_files if file.endswith(".yaml")]
        if len(cfg_files) > 0:
            cfg_file = cfg_files[0]
            file_path = os.path.join(experiment_dir, cfg_file)
        
        print("Find the latest checkpoint: ", args.checkpoint)
    print("Using config file: ", file_path)
        
    cfg = load_cfg(file_path)
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob
    if args.debug:
        cfg['env']['numEnvs'] = 34
        
    if args.eval:
        cfg['env']['numEnvs'] = 34
        cfg["env"]["maxEpisodeLength"] = 1500
        if args.checkpoint:
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
            cfg["env"]["globalStepCounter"] = checkpoint_steps
    env = create_env(cfg=cfg, args=args)
    device = env.rl_device
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
    
    num_features = 0 if args.no_feature else 1024
    encode_dim = 0 if args.no_feature else 128
    models_ppo = {}
    models_ppo["policy"] = Policy(env.observation_space, env.action_space, device, num_features=num_features, encode_dim=encode_dim, use_tanh=args.use_tanh, clip_actions=args.use_tanh, deterministic=args.eval)
    models_ppo["value"] = Value(env.observation_space, env.action_space, device, num_features=num_features, encode_dim=encode_dim)
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 24  # memory_size
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 6  # 24 * 8192 / 32768
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 5e-4
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 1.0
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0
    cfg_ppo["rewards_shaper"] = None
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
    cfg_ppo["experiment"]["write_interval"] = 24
    cfg_ppo["experiment"]["checkpoint_interval"] = 500
    cfg_ppo["experiment"]["directory"] = args.experiment_dir
    cfg_ppo["experiment"]["experiment_name"] = args.wandb_name
    cfg_ppo["experiment"]["wandb"] = args.wandb
    if args.wandb:
        cfg_ppo["experiment"]["wandb_kwargs"] = {"project": args.wandb_project, "tensorboard": False, "name": args.wandb_name}
        # cfg_ppo["experiment"]["wandb_kwargs"]["resume"] = True
        
    agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    
    cfg_trainer = {"timesteps": args.timesteps, "headless": True}
    if args.checkpoint:
        print("Resuming from checkpoint: ", args.checkpoint)
        agent.load(args.checkpoint)
        checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        if args.record_video:
            experiment_dir = args.checkpoint.split("/")[0]
            wandb_name = args.checkpoint.split("/")[1]
            cfg_trainer["video_name"] = wandb_name +"-"+str(checkpoint_steps)
            cfg_trainer["log_dir"] = experiment_dir
            cfg_trainer["record_video"] = True
        if not args.eval:
            cfg_trainer["initial_timestep"] = checkpoint_steps
            agent.set_running_mode("eval")
    
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    if args.wandb:
        import wandb
        wandb.save("data/cfg/" + cfg_file, policy="now")
        wandb.save("envs/b1z1_" + args.task[4:].lower() + ".py", policy="now")
        wandb.save("train_multistate.py", policy="now")
    if not args.eval:
        if not os.path.exists(os.path.join(args.experiment_dir, args.wandb_name, cfg_file)):
            copy_cfg(file_path, os.path.join(args.experiment_dir, args.wandb_name))
    
    return trainer
    
if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train()
    
