# LancBiO Implementation

Code repository of "LancBiO: Dynamic Lanczos-aided Bilevel Optimization via Krylov Subspace".



## Dependencies

- Ubuntu 20.04 
- Python 3.8 
- PyTorch 1.13
- CUDA 11.7
- wandb 0.16.0 (we rely on "Weights & Biases" for visualization and monitoring)


## Get Started

You can create a conda environment by simply running the following commands.

```bash
$ conda create -n lancbio_env python=3.8
$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install wandb
```

To start training, run

```bash
$ conda activate lancbio_env
$ cd ./Code/LancBiO_hyperclean_dataset    # dataset = Fashion, KMNIST, or MNIST
$ python3 main.py
```

```bash
$ conda activate lancbio_env
$ cd ./Code/LancBiO_stochastic
$ python3 main.py
```

```bash
$ conda activate lancbio_env
$ cd ./Code/SyntheticProblem
$ python3 main.py
```

## Authors

- Yan Yang (AMSS, China)



## Copyright

Copyright (C) 2025, Yan Yang, Bin Gao, Ya-xiang Yuan.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/