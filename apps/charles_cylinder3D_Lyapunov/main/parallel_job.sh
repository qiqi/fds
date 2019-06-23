#!/bin/bash
#SBATCH --job-name=fds_cylinder_CLV_finer_reso
#SBATCH --workdir=/scratch/niangxiu/fds_4CLV_finer_reso/apps/
#SBATCH --output=out
#SBATCH --error=out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16

#export PYTHONPATH=/master/home/niangxiu/.local/lib/python3.5/site-packages
#python3 drawCLV.py
#python3 draw_angles.py

export PYTHONPATH=/master/home/niangxiu/.local/lib/python2.7/site-packages
python draw_angles.py
