# Macro-Action Generator-Critic (MAGIC): Learning Macro-actions for Online POMDP Planning

This repository contains the source code for our project on learning macro-actions using the Macro-Action Generator-Critic (MAGIC) algorithm:

    @INPROCEEDINGS{Lee-RSS-21, 
        AUTHOR    = {Yiyuan Lee AND Panpan Cai AND David Hsu}, 
        TITLE     = {{MAGIC: Learning Macro-Actions for Online POMDP Planning }}, 
        BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
        YEAR      = {2021}, 
        ADDRESS   = {Virtual}, 
        MONTH     = {July}, 
        DOI       = {10.15607/RSS.2021.XVII.041} 
    } 

### Overview

The source code is split into two parts: codes written in C++ (`cpp` folder) and codes written in Python (`python` folder). The C++ codes consist of task environments, planners, and belief trackers, which are compiled into binaries. The Python codes consist of scripts to run experiments on the compiled C++ binaries.

**If you are looking for the specifics of each task (e.g. parameters, constants, dynamics)**, jump ahead to:

- [`cpp/include/core/simulations/`](cpp/include/core/simulations/) for parameters and constants
- [`cpp/src/core/simulations/`](cpp/src/core/simulations/) for task dynamics and observation models

# Getting Started

### C++

You will need to compile the C++ binaries to run the Python experiments.

##### Prerequisites
For compiling the binaries, you will need to have:
- [Boost library](https://www.boost.org/)
- [OpenCV library](https://opencv.org/)
- At least GCC-7.0
- At least CMake 3.14

#### Compiling
- `mkdir cpp/build; cd cpp/build;`
- `cmake ..`
- `make`

Ensure that the Boost and OpenCV headers and libraries are accessible during compilation (`cmake` and `make`). If installed in a custom location, you may find the CMake flag `cmake .. -DCMAKE_PREFIX_PATH=<custom location>` useful.

### Python

#### Prerequisites
To run the (virtual) experiments, you will need to have:
-  C++ binaries compiled (see previous C++ section)
- At least Python 3.6.8
- A CUDA enabled GPU
- Dependent PIP packages: `pip3 install np torch pyzmq gym tensorboard`
- Additional dependent packages: [`power_spherical`](https://github.com/nicola-decao/power_spherical)

# Running Experiments
The `python` folder contains all scripts to run experiments. It is split into multiple subfolders each serving a different purpose.

- ~~`alphabot/`: Scripts which are run on a real world robot to listen to control commands over WiFi~~ (You won't need this for the virtual experiments).
- `mdespot_handcrafted/`: Scripts to run (Macro-)DESPOT using handcrafted actions/macro-actions on our tasks.
    - `evaluate.py`: to visualize the approach.
        - e.g. `python3 evaluate.py --task=LightDark --macro-length=4`
    - `benchmark.py`: to test performance.
        - e.g. `python3 benchmark.py --task=LightDark --macro-length=4 --num-env=16`
- `mdespot_magic/`: Scripts to run DESPOT using MAGIC on our tasks.
    - `evaluate.py` to visualize the approach using a trained Generator.
      - e.g. `python3 evaluate.py --task=LightDark --macro-length=8 --model-path=../models/learned_LightDark_8 --model-index=500000`
    - `benchmark.py` to test performance using a trained Generator.
      - e.g. `python3 benchmark.py --task=LightDark --macro-length=8 --num-env=16 --models-folder=../models/learned_LightDark_8 --model-index=500000`
    - `train.py` to train both Generator + Critic.
      - e.g. `python3 train.py --task=LightDark --macro-length=8 --num-env=16 --num-iterations=500000 --output-dir=../models/learned_LightDark_8`
- `models/`: Contains the neural networks used. Also contains the trained models for each task.
- `pomcpow/`: Scripts to run POMCPOW for our tasks.
    - `valuate.py`: to visualize the approach.
        - e.g. `python3 --task=LightDark`
    - `benchmark.py`: to test performance.
        - e.g. `python3 --task=LightDark --num-env=16 `
    - `tune_params.py`: to tune the POMCPOW hyperparameters via grid search.
        - e.g. `python3 --task=LightDark --trials=30 --num-env=16`
- ~~`rwi/`: Scripts to run real world experiments~~ (You won't need this for the virtual experiments).
- `utils/`: Miscellaneous utilities and static variables.

# Running Tests

You can run tests to test our implementation of the algorithms used for comparison.

### POMCPOW

We re-implemented POMCPOW in C++, for fair comparison, following [the original paper](https://arxiv.org/abs/1709.06196). The authors' original implementation can be found [here](https://github.com/JuliaPOMDP), in Julia.

To verify our implementation, run `python3 pomcpow/benchmark.py --task=VdpTag`.

- This runs our C++ POMCPOW implementation against a C++ port of the [VdpTag task](https://github.com/zsunberg/VDPTag2.jl).
- The accumulated reward should substantially exceed the authors' Julia version on the same machine, due to optimizations by GCC.
