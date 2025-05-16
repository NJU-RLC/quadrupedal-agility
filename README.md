# Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots #

<div align="center">

[[Website]](https://nju-rlc.github.io/quadrupedal_agility/)
[[Paper]](https://arxiv.org/abs/2505.09979)
[[Video]](https://youtu.be/dUdg_KvlaLs)

</div>

<p align="center">
<img src="./images/agility.jpg" width="100%"/>
</p>

## TODO
- [x] Release motion capture data
- [x] Release BBC Training Pipeline
- [ ] Release TSC-teacher Training Pipeline
- [ ] Release TSC-student Training Pipeline
- [ ] Release EASI Training Pipeline
- [ ] Release motion retargeting pipeline
- [ ] Release deployment pipeline

## Installation ##
- Create a new python virtual environment with `python 3.8`
```bash
conda create -n agility python=3.8
conda activate agility
```
- Install `pytorch 2.3.1` with `cuda-11.8`
```bash
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 torchaudio==2.3.1+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html
```
- Install `Isaac Gym`
  - Download and install [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym)
  ```bash
    cd isaacgym/python
    pip install -e .
  ```
  - Try running an example
  ```bash
    cd examples
    python 1080_balls_of_solitude.py
  ```
  - For troubleshooting, check docs in `isaacgym/docs/install.html`
- Clone this repository and install the additional dependencies
```bash
git clone https://github.com/NJU-RLC/quadrupedal-agility.git
cd quadrupedal-agility
pip install -r requirements.txt
```

## Usage ##
### Train BBC ###
Go to `bbc/legged_gym/scripts` and run
```bash
python train.py --task go2_locomotion --experiment_idx 0 --device_id 0 --headless True
```
- Train ~200k iterations (~4 days on 3090). The trained policy is saved in `bbc/logs/go2_locomotion/0/model.pt`.
- Use `--resume --load_run 0` to start training from a saved checkpoint.

### Play BBC ###
Go to `bbc/legged_gym/scripts` and run
```bash
python play.py --task go2_locomotion --load_run 0
```
- Continuously press `W/S` for acceleration and deceleration, control steering with `A/D`, and switch behavior modes between `1-5`.

## Citation ##
If you found any part of this code useful, please consider citing:
```
@article{fu2025learning,
      title={Learning Diverse Natural Behaviors for Enhancing the Agility of Quadrupedal Robots}, 
      author={Huiqiao Fu and Haoyu Dong and Wentao Xu and Zhehao Zhou and Guizhou Deng and Kaiqiang Tang and Daoyi Dong and Chunlin Chen},
      year={2025},
      eprint={2505.09979},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.09979}
}
```
## References ##
The code is built upon the open-sourced [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs), [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [legged_gym](https://github.com/leggedrobotics/legged_gym), [MetalHead](https://github.com/inspirai/MetalHead), and [extreme-parkour](https://github.com/chengxuxin/extreme-parkour).
