from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from legged_gym.utils.helpers import get_load_path

import numpy as np
import torch
import time
import sys

np.set_printoptions(precision=3, suppress=True)

B1_FREQ = 50
B1_STEP_TIME = 1./B1_FREQ
LOW_HIGH_RATE = 5
Z1_FREQ = 500

LIN_VEL_X_CLIP = 0.15
ANG_VEL_YAW_CLIP = 0.3
ANG_VEL_PITCH_CLIP = ANG_VEL_YAW_CLIP

GAIT_WAIT_TIME = 35

class ManipLoco_Policy():
    def __init__(self, args) -> None:
        self.args = args
        self.env = None
        self.policy = None
        self.obs = None
        self.env_cfg = None
        self.init_env()
        self.init_logger()
        self.timestamp = 0

    def init_logger(self):
        logger = Logger(self.env.dt)
        robot_index = 0 # which robot is used for logging
        joint_index = 1 # which joint is used for logging
        stop_state_log = 100 # number of steps before plotting states
        stop_rew_log = self.env.max_episode_length + 1 # number of steps before print average episode rewards
        camera_position = np.array(self.env_cfg.viewer.pos, dtype=np.float64)
        camera_vel = np.array([1., 1., 0.])
        camera_direction = np.array(self.env_cfg.viewer.lookat) - np.array(self.env_cfg.viewer.pos)
        img_idx = 0

    def init_env(self):
        log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(self.args.proj_name) + self.args.exptid
        env_cfg, train_cfg = task_registry.get_cfgs(name=self.args.task)
        # override some parameters for testing
        env_cfg.env.num_envs = 1
        env_cfg.env.teleop_mode = True
        env_cfg.env.episode_length_s = 10000
        env_cfg.domain_rand.push_robots = False
        env_cfg.terrain.num_rows = 2
        env_cfg.terrain.num_cols = 3

        self.env_cfg = env_cfg

        # prepare environment
        self.env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=env_cfg)

        # initial observation
        self.obs = self.env.get_observations()

        # load policy
        train_cfg.runner.resume = True
        ppo_runner, train_cfg, checkpoint, log_pth  = task_registry.make_alg_runner(log_root = log_pth, env=self.env, name=self.args.task, args=self.args, train_cfg=train_cfg, return_log_dir=True)
        self.policy = ppo_runner.get_inference_policy(device=self.env.device, stochastic=self.args.stochastic)

        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)

        if SAVE_ACTOR_HIST_ENCODER:
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            model_file = get_load_path(log_root, load_run=self.args.load_run, checkpoint=self.args.checkpoint)
            model_name = model_file.split('/')[-1].split('.')[0]
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, train_cfg.runner.load_run, 'exported')
            os.makedirs(path, exist_ok=True)
            torch.save(ppo_runner.alg.actor_critic.actor.state_dict(), path + '/' + model_name + '_actor.pt')
            print('Saved actor to: ', path + '/' + model_name + '_actor.pt')
        
        if self.args.use_jit:
            path = os.path.join(log_pth, 'traced', self.args.exptid + "_" + str(self.args.checkpoint) + "_jit.pt")
            print("Loading jit for policy: ", path)
            self.policy = torch.jit.load(path, map_location=ppo_runner.device)

    def step(self):
        start_time = time.time()
        obs = self.obs

        if args.use_jit:
            actions = self.policy(torch.cat((obs[:, :self.env.cfg.env.num_proprio], obs[:, self.env.cfg.env.num_priv:]), dim=1))
        else:
            actions = self.policy(obs.detach(), hist_encoding=True)

        self.obs, _, rews, arm_rews, dones, infos = self.env.step(actions.detach())

        if self.timestamp % 10 == 0:
            print(self.env.ee_pos,self.env.curr_ee_goal_cart_world)
        stop_time = time.time()
        duration = stop_time - start_time
        time.sleep(max(0.02 - duration, 0))

        self.timestamp += 1

if __name__ == "__main__":
    EXPORT_POLICY = False
    SAVE_ACTOR_HIST_ENCODER = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False

    args = get_args()
    manipLoco = ManipLoco_Policy(args)
    while True:
        manipLoco.step()
