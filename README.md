# LancBiO Implementation

Code repository of "LancBiO: Dynamic Lanczos-aided Bilevel Optimization via Krylov Subspace".



## Dependencies

- Ubuntu 20.04 
- Python 3.8 
- PyTorch 1.13.1 
- CUDA 11.7
- wandb 0.16.0 (We heavily rely on Weights & Biases for visualization and monitoring)


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