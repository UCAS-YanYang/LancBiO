# LancBiO

Anonymous code repository of "LancBiO: Dynamic Lanczos-aided Bilevel Optimization via Krylov Subspace".



## Dependencies

- Ubuntu 20.04
- Python 3.8
- PyTorch 1.13.1
- CUDA 11.7
- wandb 0.16.0 (We heavily rely on Weights & Biases for visualization and monitoring)



## Data and Problem

The **data hyper-cleaning** task, conducted on the **MNIST** dataset, aims to train a classifier in a corruption scenario, where the labels of the training data are randomly altered to incorrect classification numbers at a certain probability $p$, referred to as the corruption rate. The task is formulated as follows,
![hyperclean](README/hyperclean.png)
where $L(\cdot)$ is the cross-entropy loss, $\sigma(\cdot)$ is the sigmoid function which can be viewed as the confidence for each data, and $C_r$ is a regularization parameter.



## Get Started

You can create a conda environment by simply running the following commands.

```bash
$ conda create -n lancbio_env python=3.8
$ conda activate lancbio_env
$ pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
$ pip install wandb
```

To start training, run

```bash
$ cd ./LancBiO
$ python3 main.py
```

