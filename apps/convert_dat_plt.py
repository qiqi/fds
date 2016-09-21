from __future__ import print_function

import os
import sys
import string
import subprocess
import multiprocessing
my_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(my_path, '..'))
import fds
from fds.checkpoint import *

def call_preplot(f):
	subprocess.check_call(['/hafs_x86_64/preplot', f])

k_segments = 400
m_modes = 16
for i_segment in range(298, k_segments):
    print(i_segment)
    data_files = ['segment{0:02d}_baseline'.format(i_segment)] \
               + ['segment{0:02d}_init_perturb{1:03d}'.format(i_segment, j)
                  for j in range(m_modes)]
    data_files = ['fun3d_alpha/{0}/rotated_tec_boundary.dat'.format(f)
                  for f in data_files]
    p = multiprocessing.Pool(9)
    p.map(call_preplot, data_files)

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
