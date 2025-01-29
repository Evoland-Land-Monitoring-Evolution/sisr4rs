#!/bin/bash
#SBATCH --output=slurm-logs/%j.out
#SBATCH --error=slurm-logs/%j.err
#SBATCH --partition=gpu_a100

#SBATCH --qos=gpu_all
#SBATCH -N 1                        # number of nodes ( or --nodes=1)
#SBATCH --gres=gpu:1                # number of gpus
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00             # Walltime 24h
#SBATCH --mem-per-cpu=7G           # memory per cpu
#SBATCH --account=cesbio            # MANDATORY : account (launch myaccounts to list your accounts)
#SBATCH --export=none              #  to start the job with a clean environnement and source of ~/.bashrc
cd $HOME/src/torchsisr/
module load conda
conda activate .pixi/envs/default
cd bin
PYTHONOPTIMIZE=TRUE srun --export=ALL python ./test.py "$@"
