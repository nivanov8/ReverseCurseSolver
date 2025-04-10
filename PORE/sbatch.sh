#!/bin/bash

#SBATCH --time=30:00:00
#SBATCH --partition=gpunodes
#SBATCH --nodelist=gpunode26
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/expires-2025-Apr-10/abdulbasit/run_llm.out
#SBATCH --error=/scratch/expires-2025-Apr-10/abdulbasit/run_llm.err

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_PATH=/usr/local/cuda/bin

srun python3 dataset.py