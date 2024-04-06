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
from solvers.BSA import bsa_solver

torch.set_default_dtype(torch.float64)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--training_size', type=int, default=20000)
    parser.add_argument('--validation_size', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_size', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=5000, help='K')
    parser.add_argument('--data_path', default='data/', help='The temporary data storage path')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='SOBA', choices=['LancBiO','SubBiO','AmIGO-GD','AmIGO-CG','SOBA','TTSA','stocBiO'])
    parser.add_argument('--hessian_q', type=int, default=1, help='inner iterations to update v')
    parser.add_argument('--iterations', type=int, default=1, help='inner iterations to update y')
    parser.add_argument('--outer_lr', type=float, default=100, help='step size to update x')
    parser.add_argument('--inner_lr', type=float, default=0.1, help='step size to update y')
    parser.add_argument('--eta', type=float, default=0.1, help='step size to update v')
    parser.add_argument('--noise_rate', type=float, default=0.5, help='corruption rate')                             
    parser.add_argument('--test_fre', type=int, default=30, help='frequency of data visualization')

    # LancBiO parameters
    parser.add_argument('--m', type=int, default=1)
    parser.add_argument('--dim_inc', type=float, default=1.3)
    parser.add_argument('--dim_max', type=int, default=10)
    parser.add_argument('--dim_fre', type=int, default=50)

    # parameters for TTSA
    parser.add_argument('--beta', type=float, default=100, help='beta')    
    parser.add_argument('--b', type=float, default=0.0, help='exponent b')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--a', type=float, default=0, help='exponent a')

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
    elif args.alg == 'BSA':
        bsa_solver(args, train_loader, test_loader, val_loader)

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
            args.iterations = 9
            args.hessian_q = 9
            args.batch_size = 256
        elif args.alg == 'SOBA':
            args.iterations = 1
            args.hessian_q = 1
            args.batch_size = 128
            args.epochs = 20000        
        elif args.alg == 'SubBiO':        
            args.iterations = 3
            args.batch_size = 256
        elif args.alg == 'LancBiO':
            args.iterations = 3
            args.batch_size = 256
        elif args.alg == 'TTSA':
            args.hessian_q = 9
            args.iterations = 1
            args.batch_size = 256
            args.a = 0.0
            args.b = 0.0
        elif args.alg == 'BSA':
            args.hessian_q = 9
            args.iterations = 9
            args.batch_size = 256
            args.a = 0.0
            args.b = 0.0
        elif args.alg=='AmIGO-GD':
            args.hessian_q = 9
            args.iterations = 6
            args.batch_size = 256
        elif args.alg=='AmIGO-CG':
            args.hessian_q = 9
            args.iterations = 6
            args.batch_size = 5000
        train_loader, test_loader, val_loader = get_data_loaders(args)
        train_model(args, train_loader, test_loader, val_loader)

if __name__ == '__main__':
    import wandb

    sweep_config = {
        'method': 'grid'
        }

    parameters_dict = {
        'seed':{
            'values':[1,2,3,4,5,6,7,8,9,10]#,2,3,4,5,6,7,8,9,10
            },
        'noise_rate': {
            'values': [0.5]
            },
        'alg': {
            'values': ['LancBiO','SOBA','stocBiO','AmIGO-GD','BSA','SubBiO','TTSA']#,'SubBiO','AmIGO-GD','SOBA','TTSA','stocBiO','SOBA','LancBiO','AmIGO-CG','SubBiO'
            }
        }
    sweep_config['parameters'] = parameters_dict
    pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="LancBiO_Experiments_DataClean")
    wandb.agent(sweep_id, train)
