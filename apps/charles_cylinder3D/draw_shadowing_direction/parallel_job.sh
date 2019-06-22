#!/bin/bash
#SBATCH --job-name=drawvperp_u
#SBATCH --workdir=/scratch/niangxiu/fds/apps/change_u_finer_reso
#SBATCH --output=out
#SBATCH --error=out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4

export PYTHONPATH=/master/home/niangxiu/.local/lib/python3.5/site-packages
python3 drawshadowing.py
