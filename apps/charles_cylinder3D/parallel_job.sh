#!/bin/bash
#SBATCH --job-name=fds_cylinder_change_rot_finer_reso
#SBATCH --workdir=/scratch/niangxiu/fds/apps/
#SBATCH --output=out
#SBATCH --error=out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16

#export PYTHONPATH=$PYTHONPATH:/master/home/niangxiu/.local/lib/python3.5/site-packages
export PYTHONPATH=/master/home/niangxiu/.local/lib/python3.5/site-packages
python3 charles.py
#_#SBATCH --nodelist=node8

