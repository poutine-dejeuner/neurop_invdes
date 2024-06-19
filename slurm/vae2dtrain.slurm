#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=vae2d
#SBATCH --output=%j_%x.log
#SBATCH --partition=long

CONDAENV=vae
LOGS=/home/mila/l/letournv/repos/neurop_invdes/logs
CKPTS=~/scratch/checkpoints/

mkdir -p $LOGS
mkdir -p $CKPTS

conda activate $CONDAENV

BATCH=128
STEPS=20000

srun --exclusive -n 1 -c 16 -G 1 --mem=125400mb python3 train_vae.py \
    --shape 101 91\
    --batch-size $BATCH \
    --latent-dim 2048 \
    --steps $STEPS \
    --kld-weight 0.2 --kld-weight-annealing 1e-3 2e-3 0.4 \
    --bin-weight 1.0 --bin-weight-annealing 0.0 5e-3 0.75 \
    --channels-encode 8 16 32 64 128 256 \
    --channels-decode 256 128 64 32 16 8 \
    --log-dir $LOGS \
    --checkpoint-dir $CKPTS &

wait
