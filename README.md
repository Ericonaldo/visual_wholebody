# Visual Whole-Body for Loco-Manipulation

https://wholebody-b1.github.io/

Related to paper <[Visual Whole-Body Control for Legged Loco-Manipulation](https://arxiv.org/abs/2403.16967)>

<p align="center">
<img src="./teaser.jpg" width="80%"/>
</p>

## Model learning reference

Low-level learning curves: [wandb](https://wandb.ai/ericonaldo/b1z1-low)

High-level learning curves: [wandb](https://wandb.ai/ericonaldo/b1-pick-multi-teacher)

Low-level model weights: https://drive.google.com/file/d/1KIfKu77QkrwbK-YllSWclqb6vJknGgjv/view?usp=sharing

## Set up the environment
```bash
conda create -n b1z1 python=3.8 # isaacgym requires python <=3.8
conda activate b1z1

git clone git@github.com:Ericonaldo/visual_whole_body.git

cd visual_whole_body

pip install torch torchvision torchaudio

cd third_party/isaacgym/python && pip install -e .

cd ../..
cd rsl_rl && pip install -e .

cd ..
cd skrl && pip install -e .

cd ../..
cd low-level && pip install -e .

pip install numpy pydelatin tqdm imageio-ffmpeg opencv-python wandb
```

## Structure

- `high-level`: codes and environments related to the visuomotor high-level policy, task-relevant

- `low-level`: codes and environments related to the general low-level controller for the quadruped and the arm, the only task is to learn to walk while tracking the target ee pose and the robot velocities.

Detailed code structures can be found in these directories.

## How to work (roughly)

- Train a low-level policy using codes and follow the descriptions in `low-level`

- Put the low-level policy checkpoint into somewhere.

- Train the high-level policy using codes and follow the descriptions in `high-level`, while assigning the low-level model in the config yaml file.

## Acknowledgements (third-party dependencies)

- [isaacgym](https://developer.nvidia.com/isaac-gym)
- [legged_gym](https://github.com/leggedrobotics/legged_gym)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [skrl](https://github.com/Toni-SM/skrl)

## Codebase Contributions

- [Minghuan Liu](https://minghuanliu.com) made efforts on improving the training efficiency, reward engineering, filling sim2real gaps, and reach expected behaviors, while cleaning and integrating the whole codebase for simplicity.
- [Zixuan Chen](https://zixuan417.github.io) initialized the code base and made early progress on reward design, training, testing, and sim2real transferring, along with some baselines.
- [Xuxin Cheng](https://chengxuxin.github.io/) shared a lot of domain knowledge and reward experience on locomotion and low-level policy training, and helped debug the code.
- [Xuanbin Peng](https://github.com/xuanbinpeng) cleaned and refactored the low-level codebase to improve the readability while also finetuned the reward function for a stable walking.
- [Yandong Ji](https://yandongji.github.io/) provided several suggestions and helped debug the code.

## Citation
If you find the code base helpful, consider to cite
```
@article{liu2024visual,
    title={Visual Whole-Body Control for Legged Loco-Manipulation},
    author={Liu, Minghuan and Chen, Zixuan and Cheng, Xuxin and Ji, Yandong and Yang, Ruihan and Wang, Xiaolong},
    journal={arXiv preprint arXiv:2403.16967},
    year={2024}
}
```
