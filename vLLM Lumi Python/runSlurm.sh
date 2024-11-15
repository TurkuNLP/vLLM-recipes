#!/bin/bash
#SBATCH --job-name=llamaRun
#SBATCH --account=project_462000642
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node=8
#SBATCH --mem=250G
#SBATCH --time=1:00:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch


gpu-energy --save

srun python3 vLLM_inference.py

gpu-energy --diff