#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH -q qos_gpu-dev
#SBATCH -A fwp@a100
#SBATCH -C a100
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00             # Walltime 30 min
#SBATCH --hint=nomultithread

cd $WORK/src/torchsisr/

module purge
module load python/3.11.5
conda activate .pixi/envs/default

cd bin
PYTHONOPTIMIZE=TRUE python -u ./test.py "$@"
