from __future__ import print_function

import os
import sys
my_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(my_path)
from fun3d import *

def most_recent_checkpoint(m):
    filter_func = lambda f: f.startswith('m{0}_segment'.format(m))
    files = sorted(filter(filter_func, os.listdir(BASE_PATH)))
    if len(files):
        return load_checkpoint(os.path.join(BASE_PATH, files[-1]))

cp = most_recent_checkpoint(M_MODES)
verify_checkpoint(cp)

L = cp.lss.lyapunov_exponents()

def exp_mean(x):
    n = len(x)
    w = 1 - exp(range(1,n+1) / sqrt(n))
    x = array(x)
    w = w.reshape([-1] + [1] * (x.ndim - 1))
    return (x * w).sum(0) / w.sum()

n_exp = 5
print(' '.join(['segs'] + ['Lyap exp {:<2d}'.format(i) for i in range(n_exp)]))
for i in range(1, len(L)):
    print(' '.join(['{:<4d}'.format(i)] +
                   ['{:<+11.2e}'.format(lam) for lam in exp_mean(L[:i, :5])]))
