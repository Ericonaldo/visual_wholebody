# Train a task-relevant high-level policy

This code base only includes the task of picking multiple objects.

## Code structure
`data` contains assets and config files.

`envs` contains environment related codes.

`learning` contains dagger training codes (student distillation)

`modules` contains some network structures.

`utils` contains arguments, wrappers, and low-level policy definitions.

## Train

1. Environments: 
   **Picking multiple objects**: [b1_z1pickmulti.py](./envs/b1_z1pickmulti.py) is for walking and picking policy, supporting floating base (fixed and varied body heights).

2. To train the state-based teacher, using [train_multistate.py](./train_multistate.py) with the config [b1z1_pickmulti.yaml](./data/cfg/b1z1_pickmulti.yaml) (remember to determine the pre-trained low-level policy path in the config file):
```bash
python train_multistate.py --rl_device "cuda:0" --sim_device "cuda:0" --timesteps 60000 --headless --task B1Z1PickMulti --experiment_dir b1-pick-multi-teacher --wandb --wandb_project "b1-pick-multi-teacher" --wandb_name "some descriptions" --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick
```
Arguments explanation:

`--task` should be the full name of the environment, with every first letter of each word capitalized. 

`--timesteps` total training steps. 

`--experiment_dir` is the name of the directory where the running is saved.

`--wandb`, then the training will be logged to wandb. You can omit this argument if you don't want to use wandb or when debugging. 

`--wandb_project` is the name of the project in wandb. 

`--wandb_name` is the name of the run in wandb, which is also the name of this run in experiment_dir.

`--roboinfo` means the high-level policy reads low-level proprioception states.

`--observe_gait_commands` is for tracking specific gait commands and learning the trotting behavior.

`--small_value_set_zero` is to clip the small command velocity into 0, which should be the same as the low-level policy.

`--rand_control` randomizes the high-low control frequency.

`--stop_pick` enforces the robot to stop when the gripper is closing.

For playing the teacher policy, using [play_multistate.py](./play_multistate.py):
```bash
python play_multistate.py --task B1Z1PickMulti --checkpoint "(specify the path)" # --(same arguments as training)
```
It should be a maximum of 60000 timesteps for successful teacher policy training.


3. To train the vision-based student policy, use [train_multi_bc_deter.py](./train_multi_bc_deter.py) with the config [b1z1_pickmulti.yaml](./data/cfg/b1z1_pickmulti.yaml)
```bash
python train_multi_bc_deter.py --headless --task B1Z1PickMulti --rl_device "cuda:0" --sim_device "cuda:0" --timesteps 60000 --experiment_dir "b1-pick-multi-stu" --wandb --wandb_project "b1-pick-multi-stu" --wandb_name "checkpoint dir path" --teacher_ckpt_path "teacher checkpoint path" --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --stop_pick
```
Arguments are similar to those above.

For playing the trained student policy, using [play_multi_bc_deter.py](./play_multi_bc_deter.py):
```bash
python play_multi_bc_deter.py --task B1Z1PickMulti --checkpoint "(specify the path)" # --(same arguments as training)
```
If you don't specify `--num_envs`, it will use 34 by default (only for this script).
It should be a maximum of 60000 timesteps for successful student policy training.

## Others
[test_pointcloud.py](./test_pointcloud.py) can be use for checking the pointcloud of the objects.

[train_multistate_asym.py](./train_multistate_asym.py) is a try of using asymetric PPO for training the high-level policy (i.e, vision-based policy and privilaged value function), it is training inefficient and is not comparable to the teacher-student as it cannot parallel too many environments due to the depth images consumption.
