import torch
import wandb
import argparse
import numpy as np
import random

import os
from pprint import pprint
from distutils.util import strtobool

from problems.test_problem import Test_Problem
from solvers.LancBiO import lancbio
from solvers.AmIGO_GD import amigo_gd
from solvers.AmIGO_CG import amigo_cg
from solvers.SOBA import soba
from solvers.stocBiO import stocbio
from solvers.BSA import bsa
from solvers.TTSA import ttsa
from solvers.SubBiO import subbio

torch.set_default_dtype(torch.float64)

def parse_args():
    parser = argparse.ArgumentParser()
    # experiments args
    parser.add_argument("--wandb-project-name", type=str, default="LancBiO_Sythetic_Problem",
        help="the wandb's project name")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_fre', type=int, default=50, help='test frequency')
    
    # test problem settings
    parser.add_argument('--n', default=10000, type=int, help='problem dimension')
    parser.add_argument('--gamma', default=0.5, type=float, help='problem parameter')
    parser.add_argument('--condition', default=100000, type=float, help='problem parameter')

    # algorithm settings
    parser.add_argument('--alg', type=str, default='LancBiO', choices=['stocBiO', 'LancBiO','AmIGO-CG','AmIGO-GD','SOBA','TTSA','BSA','SubBiO'])
    parser.add_argument('--K', default=10000, type=int, help='outer update iterations')
    parser.add_argument('--T', default=5, type=int, help='inner update iterations')
    parser.add_argument('--hessian_q', type=int, default=5, help='number of steps to approximate hessian')
    parser.add_argument('--lam', default=1, type=float, help='stepsize for x')
    parser.add_argument('--theta', default=0.00001, type=float, help='stepsize for y')
    parser.add_argument('--eps', default=1e-10, type=float, help='eps for hyper-gradient check')

    # step size decay
    parser.add_argument('--min_lam', default=0.0, type=float, help='min step size of x')
    parser.add_argument('--b', default=0.0, type=float, help='step size decay of x')
    parser.add_argument('--min_theta', default=0.0, type=float, help='min step size of x')
    parser.add_argument('--a', default=0.0, type=float, help='step size decay of y')

    # LancBiO dimension parameters
    parser.add_argument('--m', type=int, default=1, help='initial subspace dimention')
    parser.add_argument('--dim_max', type=int, default=15, help='max subspace dimention')
    parser.add_argument('--dim_inc', type=float, default=1.1, help='subspace dimention increasing rate')
    parser.add_argument('--dim_fre', type=int, default=20, help='subspace dimention increasing frequency')

    args = parser.parse_args()
    return args
    


def train_model(args):
    P = Test_Problem(args)

    if args.alg == 'LancBiO':
        lancbio(args,P)
    if args.alg == 'AmIGO-GD':
       amigo_gd(args,P)
    if args.alg == 'AmIGO-CG':
        amigo_cg(args,P)
    if args.alg == 'SOBA':
        soba(args,P)
    if args.alg == 'stocBiO':
        stocbio(args,P)
    if args.alg == 'TTSA':
        ttsa(args,P)
    if args.alg == 'BSA':
        bsa(args,P)
    if args.alg == 'SubBiO':
        subbio(args,P)


def train(config=None):
    args = parse_args()
    with wandb.init(config=config, save_code=True):
        config = wandb.config
        args.seed = config.seed
        args.alg = config.alg
        
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if args.alg == 'stocBiO':
            args.T = 10
            args.hessian_q = 10

        elif args.alg == 'SOBA':
            args.T = 1       
            args.hessian_q = 1
            args.K = 30000

        elif args.alg == 'SubBiO':        
            args.T = 5

        elif args.alg == 'LancBiO':
            args.m = 1              # initial subspace dimension
            args.dim_inc = 1.1      # incresing rate
            args.dim_max = 80       # subspace dimension
            args.dim_fre = 20       # increase each 20 iterations
            args.T = 5

        elif args.alg == 'BSA':
            args.min_lam = 0.01
            args.a = 0.0
            args.b = 0.5
            args.hessian_q = 10
            args.T = 10

        elif args.alg == 'TTSA':
            args.min_lam = 0.01
            args.a = 0.0
            args.b = 0.4
            args.hessian_q = 10
            args.T= 1

        elif args.alg=='AmIGO-GD' or args.alg=='AmIGO-CG':
            args.T = 5
            # args.hessian_q = config.hessian_q
            args.hessian_q = 5

        train_model(args)

if __name__ == '__main__':
    import wandb
    sweep_config = {
        'method': 'grid'
        }

    parameters_dict = {
        'seed':{
            'values':[1,2,3,4,5,6,7,8,9,10]#,1,2,3,4,5,6,7,8,9,10
            },
        'alg': {
            'values': ['LancBiO','AmIGO-CG','stocBiO','AmIGO-GD','SOBA','BSA','AmIGO-GD','SubBiO']
            }
        }
    sweep_config['parameters'] = parameters_dict
    pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="Sweeps_Synthetic_Problems")
    wandb.agent(sweep_id, train)

