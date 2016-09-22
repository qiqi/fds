from __future__ import print_function

import os
import sys
import string
import subprocess
my_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(my_path, '..'))
import fds
from fds.checkpoint import *

cp = load_last_checkpoint('fun3d_alpha', 16)
verify_checkpoint(cp)

template = string.Template(open('tecplot.mcr.template').read())

pbs_header = '''#!/bin/csh
#PBS -A AFVAW39842SAM
#PBS -l walltime=23:59:00
#PBS -l select=1:ncpus=4:mpiprocs=4
#PBS -q standard
#PBS -N tecplot
#PBS -j oe

cd $PBS_O_WORKDIR

'''

vectors = cp.lss.lyapunov_covariant_vectors()
pbs_file = 'job.pbs'
f_pbs = open(pbs_file, 'wt')
f_pbs.write(pbs_header)
for i_segment in range(100, 150):
    for i_vector, vector in enumerate(vectors[:4]):
        v = vector[i_segment]
        data_files = ['segment{0:02d}_baseline'.format(i_segment)] \
                   + ['segment{0:02d}_init_perturb{1:03d}'.format(i_segment, j)
                      for j in range(len(v))]
        data_files = ['"fun3d_alpha/{0}/rotated_tec_boundary.plt"'.format(f)
                      for f in data_files]
        data_files = ' '.join(data_files)

        deleted_zones = ','.join(['{0}-{1}'.format(i,i+2)
                                  for i in range(1,4*len(v)+2,4)])

        equation_du = '{du} = ' \
                    + ' + '.join([
                          '{0} * ({{u}}[{1}] - {{u}}[1])'.format(v[i], i+2)
                          for i in range(len(v))])
        equation_dp = '{dp} = ' \
                    + ' + '.join([
                          '{0} * ({{p}}[{1}] - {{p}}[1])'.format(v[i], i+2)
                          for i in range(len(v))])
                           
        file_suffix = 'vector_{0}_segment_{1}'.format(i_vector, i_segment)

        macro = template.substitute(
           DATA_FILES=data_files,
           DELETED_ZONES=deleted_zones,
           EQUATION_DU=equation_du,
           EQUATION_DP=equation_dp,
           FILE_SUFFIX=file_suffix)
        mcr_file = 'tecplot_macros/tecplot_{0}_{1}.mcr'.format(i_vector, i_segment)
        with open(mcr_file, 'wt') as f:
            f.write(macro)
        png_file = 'png/du_far_vector_{0}_segment_{1}.png'.format(i_vector, i_segment)
        f_pbs.write('if ( ! -f ' + png_file + ' ) tec360 -mesa -b -p ' + mcr_file + ' & \n')
        # subprocess.check_call(
        #     ['/hafs_x86_64/tec360', '-mesa', '-b', '-p', 'tecplot.mcr'])
    f_pbs.write('wait\n\n')
f_pbs.close()
         
# print(L.shape)
# 
# def exp_mean(x):
#     n = len(x)
#     w = 1 - exp(range(1,n+1) / sqrt(n))
#     x = array(x)
#     w = w.reshape([-1] + [1] * (x.ndim - 1))
#     return (x * w).sum(0) / w.sum()
# 
# n_exp = 5
# print(' '.join(['segs'] + ['Lyap exp {:<2d}'.format(i) for i in range(n_exp)]))
# for i in range(1, len(L)):
#     print(' '.join(['{:<4d}'.format(i)] +
#                    ['{:<+11.2e}'.format(lam) for lam in exp_mean(L[:i, :5])]))
