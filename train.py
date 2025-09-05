import os
import torch
import argparse
import shutil
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import os.path as osp
from omegaconf import OmegaConf
import torch.utils
from tqdm import tqdm
from datetime import timedelta
from einops import rearrange
from train_worker import train_worker
from threadpoolctl import threadpool_limits
import numpy as np
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--master_addr', type=str, default='localhost', help='Master address for distributed training')
    parser.add_argument('--master_port', type=int, default=12345, help='Master port for distributed training')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes for distributed training')
    parser.add_argument('--node_rank', type=int, default=0, help='Rank of the current node')
    args = parser.parse_args()
    return args

def seed_anything(seed: int = 42):
    """
    Set seed for reproducibility in deep learning experiments.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)                        # Python random
    np.random.seed(seed)                     # NumPy
    os.environ['PYTHONHASHSEED'] = str(seed) # Python hash seed
    torch.manual_seed(seed)                  # PyTorch CPU
    torch.cuda.manual_seed(seed)             # PyTorch GPU
    torch.cuda.manual_seed_all(seed)         # All GPUs

    # cuDNN settings for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_environment(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your setup.")
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} does not exist.")
    
    config = OmegaConf.load(args.config)
    config.update(args.__dict__)
    # if 'seed' in config:
    if config.get('seed') is not None:
        seed_anything(config.seed)
    
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    return config  


def main():
    args = parse_args()
    config = setup_environment(args)
    with threadpool_limits(limits=10, user_api="blas"):
        mp.spawn(train_worker, nprocs=config.gpus, args=(config,))

if __name__ == "__main__":
    mp.freeze_support() 
    main()



