import argparse
import os
from pathlib import Path

# import pytorch_lightning as pl
# from lightning_fabric.utilities.seed import seed_everything
from easydict import EasyDict as edict
# from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from fno_field_prediction.data import BlobData
from fno_field_prediction.models import VAE2d, VAE3d

args = edict({'seed': 912374122, 
             'latent_dim': 2048, 
             'channels_encode': [8, 16, 32, 64, 128, 256], 
             'channels_decode': [256, 128, 64, 32, 16, 8], 
             'kld_weight': 0.2, 
             'kld_weight_annealing': None, 
             'bin_cutoff': 3.0, 'bin_weight': 0.0, 
             'bin_weight_annealing': None, 
             'shape': [128, 128], 
             'sigma': 12, 
             'batch_size': 64, 
             'steps': 20000, 
             'lr': 0.0002, 
             'weight_decay': 1e-06, 
             'name': None, 
             'group': None, 
             'accelerator': 'auto', 
             'devices': 1, 
             'num_nodes': 1, 
             'num_workers': 0, 
             'strategy': None, 
             'dev': False, 
             'checkpoint_dir': None})


module = VAE3d if len(args.shape) == 3 else VAE2d
model = module(
    args.latent_dim,
    args.channels_encode,
    args.channels_decode,
    input_dim=args.shape[0],
    output_dim=args.shape[0],
    kld_weight=args.kld_weight,
    kld_weight_annealing=args.kld_weight_annealing,
    bin_weight=args.bin_weight,
    bin_cutoff=args.bin_cutoff,
    bin_weight_annealing=args.bin_weight_annealing,
    lr=args.lr,
    weight_decay=args.weight_decay,
    steps=args.steps,
)

PATH = os.path.expanduser('~/scratch/checkpoints/128128/last.ckpt')
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['state_dict'])


deco = model.decoder
deco


z = torch.rand(4,2048)

# breakpoint()
print(deco(z))


