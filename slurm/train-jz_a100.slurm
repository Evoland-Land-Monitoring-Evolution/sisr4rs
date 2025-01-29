#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --signal=SIGUSR1@90
#SBATCH -A fwp@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00             # Walltime 24h
#SBATCH --hint=nomultithread

cd $WORK/src/torchsisr/

module purge
module load python/3.11.5
conda activate .pixi/envs/default

cd bin
PYTHONOPTIMIZE=TRUE srun python -u ./train.py "$@"
