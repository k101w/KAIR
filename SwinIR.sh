#!/bin/bash

#SBATCH --job-name=SwinIR
#SBATCH --output=job.%j.out
#SBATCH --error=job.%j.err
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --partition=all-shared-gpu-preemp
#SBATCH --gres=gpu:3

module purge

module load openmpi
echo "== This is the scripting step! =="
export PYTHONVERBOSE=1
torchrun --nproc_per_node=8 --master_port=1234 main_train_psnr.py -opt options/swinir/train_swinir_sr_realworld_psnr.json  --dist True --launcher pytorch
echo "== End of Job =="