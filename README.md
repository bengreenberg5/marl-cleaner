### Overview

Custom implementation of the Cleaner game originally described [here](https://github.com/Bigpig4396/Multi-Agent-Reinforcement-Learning-Environment/blob/master/env_Cleaner/Cleaner.pdf).

![](https://raw.githubusercontent.com/bengreenberg5/zsc-cleaner/master/cleaner_simple.png)

A video of self-play can be seen [here](https://www.youtube.com/watch?v=NcSvX9B6ACs).

Cleaner is a simple cooperative game meant to illustrate the difficulty of [zero-shot coordination](https://arxiv.org/abs/2106.06613v1). Multiple agents travel around a rectangular grid, turning "dirty" squares into "clean" squares by visiting them. The game is similar to Pac-Man -- with multiple players and no monsters.

Each time step, agents can move 1 space up, down, left, or right; they receive a shared reward of 1 for each square cleaned, and incur a small time penalty. The episode ends after a fixed number of time steps, or when there are no remaining dirty squares. Once an agent visits a square, it is clean for the remainder of the episode. Agents can occupy the same square concurrently, but cannot receive double rewards for doing so.

This work was completed as part of SERI's 2021 Summer Research Program. A write-up describing the experiments can be found [here](https://drive.google.com/file/d/1bb4MJENEPSIdV0O4P_QrhDJC8UfPVZU5/view?usp=sharing).

### Installation

Create a new virtual environment and install packages with `pip install -r requirements.txt`.

In order to render videos of episodes, you may have to install [ffmpeg](). If you don't want to render videos, you can run training with the `--no_record` flag.

### Repo Overview

- `cleaner/`: Implementation of environment
    - `cleaner_game.py`: Base class for game logic
    - `cleaner_env.py`: Wrapper around `CleanerGame` for gym API
    - `utils.py`: Utility functions for training and evaluation
- `configs/`: Collection of configuration files for running experiments
    - `simple_2.yaml`: 11x11 board with 2 agents
    - `simple_4.yaml`: 11x11 board with 4 agents
    - `simple_8.yaml`: 11x11 board with 8 agents
    - `ring_4.yaml`: 11x11 board with an inscribed cross and 4 agents
    - `halls.yaml`: 8x9 board with 6 "hallways"
    - `simple_4_lava.yaml`: like `simple_4` but visiting a clean square incurs a penalty
- `videos/`: Miscellaneous episode recordings
- `run.py`: Main class for running training
- `evaluation.ipynb`: Jupyter notebook for evaluating self-play and cross-play scores. The file is too large to view in GitHub
- `test_cleaner.py`: Environment tests

### Usage

After installing the requirements, you can run training with:

```python run.py --name test_run --config simple_2```

Arguments:
- `--name <NAME>`: Experiment name; results will appear in a subdirectory with this name
- `--config <CONFIG>`: Name of the config file (e.g. to use `configs/simple_4.yaml` run with `--config simple_4`
- `--policy <POLICY>`: Which RL algorithm to use (default: "ppo")
- `--training_iters <NUM>`: Number of training iterations
- `--seed <SEED>`: Random seed for Ray workers
- `--homogeneous`: Centrally train one policy for all agents
- `--random_start`: Randomly initialize the starting positions each episode
- `--no_record`: Don't save video in evaluation
- `--checkpoint_freq <FREQ>`: How many training iterations between trainer checkpoints
- `--eval_freq <FREQ>`: How many training iterations between evaluations

