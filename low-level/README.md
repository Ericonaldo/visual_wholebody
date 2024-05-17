# Training a universal low-level policy

## Code structure
`legged_gym/envs` contains environment related codes.

`envs` contains environment related codes.

`learning` contains dagger training codes (student distillation)

`modules` contains some network structures.

`utils` contains arguments, wrappers and low-level policy definition.

## Train
The environment related code is `legged_gym/legged_gym/envs/manip_loco/manip_loco.py`, and the related config for b1z1 hardware is in `legged_gym/legged_gym/envs/b1z1/b1z1_config.py`.

```bash
python train.py --headless --exptid SOME_YOUR_DESCRIPTION --proj_name b1z1-low --task b1z1 --sim_device cuda:0 --rl_device cuda:0
```
- `--debug` disables wandb and set a small number of envs for faster execution.
- `--headless` disables rendering, typically used when you train model.
- `--proj_name` the folder containing all your logs and wandb project name. `manip-loco` as default.

Check `legged_gym/legged_gym/utils/helpers.py` for all command line args.

## Play
Only need to specify `--exptid`. The parser will automatically find corresponding runs.
```bash
python play.py --exptid 013X-1X-samplelow --task b1z1 --proj_name b1z1-low --checkpoint 64000
```
Use `--sim_device cpu --rl_device cpu` in case no enough gpu memory.

## Suggestions
To choose a good low-level policy that can be further used for training the high-level policy, we suggest you deploy the low-level policy first, and see if it goes well before training a high-level policy.