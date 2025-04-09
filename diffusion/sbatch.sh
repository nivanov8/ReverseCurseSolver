#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode19
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-10/abdulbasit/diffusion.out
#SBATCH --error=/scratch/expires-2025-Apr-10/abdulbasit/diffusion.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 -m diffusion.model