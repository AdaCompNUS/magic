# Macro-Action Generator-Critic (MAGIC): Learning Macro-actions for Online POMDP Planning

This repository contains the source code for our project on learning macro-actions using the Macro-Action Generator-Critic (MAGIC) algorithm.

# Overview

The source code is split into two parts: codes written in C++ (`cpp` folder) and codes written in Python (`python` folder). The C++ codes consist of task environments, planners, and belief trackers, which are built into binaries. The Python codes consist of scripts to run experiments on the compiled C++ binaries.

**If you are looking for the specifics of each task (e.g. parameters, constants, dynamics)**, jump ahead to:
- [`cpp/include/core/simulations/`](cpp/include/core/simulations/) for parameters and constants
- [`cpp/src/core/simulations/`](cpp/src/core/simulations/) for task dynamics and observation models

# Getting Started

You will need to build the C++ binaries to run the Python experiments.

## C++

### Prerequisites
For building the libraries, you will need to have:
- [Boost library](https://www.boost.org/)
- [OpenCV library](https://opencv.org/)
- At least GCC-7.0
- At least CMake 3.14

### Building
- `mkdir cpp/build; cd cpp/build;`
- `cmake ..`
- `make`

Ensure that the Boost and OpenCV headers and libraries are accessible during compilation (`cmake` and `make`). If installed in a custom location, you may find the CMake flag `cmake .. -DCMAKE_PREFIX_PATH=<custom location>` useful.

## Python

### Prerequisites
To run the (virtual) experiments, you will need to have:
- At least Python 3.6.8
- A CUDA enabled GPU
- Dependent PIP packages: `pip3 install np torch pyzmq gym tensorboard`
- Additional dependent packages: [`power_spherical`](https://github.com/nicola-decao/power_spherical)

## Running experiments
The `python` folder is split into multiple `subfolders` each serving a different purpose.

- `alphabot/`: Scripts which are run on a real world robot to listen to control commands over WiFi (You won't need this for the virtual experiments).
- `gymenvs/`: Contains OpenAI Gym wrapper environments for the C++ environment binaries for our tasks.
- `mdespot_handcrafted/`: Scripts to run DESPOT using handcrafted actions/macro-actions on our tasks.
    - `benchmark.py`: to test performance.
    - `evaluate.py`: to visualize the approach.
- `mdespot_magic/`: Scripts to run DESPOT using MAGIC on our tasks.
    - `train.py` to train a Parameters-Net.
    - `evaluate.py` to visualize the approach using a trained Parameters-Net.
    - `benchmark.py` to test performance using a trained Parameters-Net.
- `models/`: Contains the neural networks used. Also contains the trained models for each task.
- `pomcpow/`: Scripts to run POMCPOW for our tasks.
    - `benchmark.py`: to test performance.
    - `evaluate.py`: to visualize the approach.
    - `tune_params.py`: to tune the POMCPOW hyperparameters via grid search.
- `rwi/`: Scripts to run real world experiments (You won't need this for the virtual experiments).
- `sac/`: Scripts to run SAC for our tasks.
    - This folder is originally from `https://github.com/pranz24/pytorch-soft-actor-critic`. We adapt it to work with our models and environments.
    - `benchmark.py`: to test performance.
    - `main.py`: to train.
- `utils/`: Miscellaneous utilities and static variables.
