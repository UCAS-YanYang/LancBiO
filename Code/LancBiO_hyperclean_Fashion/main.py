import numpy as np
import torch
import argparse
import random
from distutils.util import strtobool
from pprint import pprint
from utils.dataset import get_data_loaders

from solvers.LancBiO import lancbio_solver
from solvers.SubBiO import subbio_solver
from solvers.AmIGO_GD import amigo_gd_solver
from solvers.AmIGO_CG import amigo_cg_solver
from solvers.SOBA import soba_solver
from solvers.TTSA import ttsa_solver
from solvers.stocBiO import stocbio_solver
from solvers.STABLE import stable_solver
from solvers.BSA import bsa_solver
from solvers.F2SA import f2sa_solver
from solvers.HJFBiO import hjfbio_solver

torch.set_default_dtype(torch.float64)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--training_size', type=int, default=5000)
    parser.add_argument('--validation_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=3000, help='K')
    parser.add_argument('--data_path', default='data/', help='The temporary data storage path')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='SOBA', choices=['LancBiO','SubBiO','AmIGO-GD','AmIGO-CG','SOBA','TTSA','stocBiO','STABLE'])
    parser.add_argument('--hessian_q', type=int, default=1, help='inner iterations to update v')
    parser.add_argument('--iterations', type=int, default=1, help='inner iterations to update y')
    parser.add_argument('--outer_lr', type=float, default=500, help='step size to update x')
    parser.add_argument('--inner_lr', type=float, default=0.1, help='step size to update y')
    parser.add_argument('--eta', type=float, default=0.05, help='step size to update v')
    parser.add_argument('--noise_rate', type=float, default=0.8, help='corruption rate')                             
    parser.add_argument('--test_fre', type=int, default=30, help='frequency of data visualization')

    # LancBiO parameters
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--dim_inc', type=float, default=1.3)
    parser.add_argument('--dim_max', type=int, default=10)
    parser.add_argument('--dim_fre', type=int, default=50)

    # parameters for TTSA
    parser.add_argument('--beta', type=float, default=500, help='beta')    
    parser.add_argument('--b', type=float, default=0, help='exponent b')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--a', type=float, default=0, help='exponent a')

    # parameters for F2SA
    parser.add_argument('--lambda0', type=float, default=1, help='penalty')    
    parser.add_argument('--delta_lambda', type=float, default=0.1, help='delta_lambda')
    parser.add_argument('--ratio', type=float, default=1, help='outer-inner step-sizes ration')

    # parameters for HJFBiO
    parser.add_argument('--delta', type=float, default=0.00001, help='parameter for finite-difference technique')

    args = parser.parse_args()
    return args

def train_model(args, train_loader, test_loader, val_loader):
    if args.alg=='LancBiO':
        lancbio_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='SubBiO':
        subbio_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='AmIGO-GD':
        amigo_gd_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='AmIGO-CG':
        amigo_cg_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='SOBA':
        soba_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='TTSA':
        ttsa_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='stocBiO':
        stocbio_solver(args, train_loader, test_loader, val_loader)
    elif args.alg=='STABLE':
        stable_solver(args, train_loader, test_loader, val_loader)
    elif args.alg == 'BSA':
        bsa_solver(args, train_loader, test_loader, val_loader)
    elif args.alg == 'F2SA':
        f2sa_solver(args, train_loader, test_loader, val_loader)
    elif args.alg == 'HJFBiO':
        hjfbio_solver(args, train_loader, test_loader, val_loader)



def train(config=None):
    args = parse_args()
    with wandb.init(config=config, save_code=True):
        config = wandb.config
        args.seed = config.seed
        args.alg = config.alg
        args.noise_rate = config.noise_rate
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        if args.alg == 'stocBiO':
            args.iterations = 10
            args.hessian_q = 10
        elif args.alg == 'SOBA':
            args.iterations = 1   
            args.epochs = 6000
            args.test_fre = 50        
        elif args.alg == 'SubBiO':        
            args.iterations = 5            
        elif args.alg == 'LancBiO':
            args.iterations = 5
        elif args.alg == 'STABLE':
            # for STABLE we adopt larger Q steps approximation instead of directly using Hessian inverse
            args.hessian_q = 20
            # args.iterations = 5 enjoys better performance
            args.iterations = 5
        elif args.alg == 'TTSA':
            args.hessian_q = 10
            args.iterations = 1
        elif args.alg == 'BSA':
            args.hessian_q = 10
            args.iterations = 10
        elif args.alg=='AmIGO-GD' or args.alg=='AmIGO-CG':
            args.hessian_q = 5
            args.iterations = 10
            if args.alg=='AmIGO-CG':
                args.outer_lr = 100
        elif args.alg == 'F2SA':
            args.iterations = 10
            args.outer_lr = 10 # when outer_l>10, F2SA becomes unstable
        elif args.alg == 'HJFBiO':
            args.iterations = 1
            args.epochs = 6000
            args.test_fre = 50
        

        train_loader, test_loader, val_loader = get_data_loaders(args)
        train_model(args, train_loader, test_loader, val_loader)

if __name__ == '__main__':
    import wandb

    sweep_config = {
        'method': 'grid'
        }

    parameters_dict = {
        'seed':{
            'values':[1,2,3,4,5,6,7,8,9,10]#,2
            },
        'noise_rate': {
            'values': [0.5,0.8]
            },
        'alg': {
            'values': ['F2SA']#,''STABLE''SOBA',,'AmIGO-CG','SubBiO','LancBiO','F2SA','SubBiO','AmIGO-CG','AmIGO-GD','SOBA','AmIGO-CG','F2SA','SOBA'
            }
        }
    sweep_config['parameters'] = parameters_dict
    pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="LancBiO_DataClean_Fashion")   
    wandb.agent(sweep_id, train)
