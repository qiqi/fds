#!/bin/bash
#SBATCH --job-name=fds_cylinder
#SBATCH --workdir=/scratch/niangxiu/fds/apps/
#SBATCH --output=out.out
#SBATCH --error=out.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16

#export PYTHONPATH=$PYTHONPATH:/master/home/niangxiu/.local/lib/python3.5/site-packages
export PYTHONPATH=/master/home/niangxiu/.local/lib/python3.5/site-packages
python3 charles.py
#_#SBATCH --nodelist=node8

