from __future__ import division

import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

from fds.timeseries import windowed_mean_weights, windowed_mean
from fds.segment import run_segment, trapez_mean

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *

solver_path = os.path.join(my_path, 'solvers', 'lorenz')
solver = os.path.join(solver_path, 'solver')
adj_solver = os.path.join(solver_path, 'adjoint')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

def solve(u, s, nsteps):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray([10, 28., 8./3, s], dtype='>d').tobytes())
    call([solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'objective.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, J

def adjoint(u, s, ua, nsteps):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray([10, s, 8./3], dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'adj-input.bin'), 'wb') as f:
        f.write(asarray(au, dtype='>d').tobytes())
    call([adj_solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'adj-output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'dJds.bin'), 'rb') as f:
        dJds = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, dJds

if __name__ == '__main__':
    m = 2
    cp = shadowing(solve, u0, 28, m, 3, 1000, 5000, return_checkpoint=True)

    _, _, _, lss, G_lss, g_lss, J, G_dil, g_dil = cp
    J = np.array(J)
    steps_per_segment = J.shape[1]
    dJ = trapez_mean(J.mean(0), 0) - J[:,-1]
    assert dJ.ndim == 2 and dJ.shape[1] == 1

    win = windowed_mean_weights(dJ.shape[0])
    g_lss_adj = win[:,newaxis]
    alpha_adj_lss = win[:,newaxis] * np.array(G_lss)[:,:,0]

    dil_adj = win * ravel(dJ)
    g_dil_adj = dil_adj / steps_per_segment
    alpha_adj_dil = dil_adj[:,newaxis] * G_dil / steps_per_segment

    alpha_adj = alpha_adj_lss + alpha_adj_dil
    b_adj = lss.adjoint(alpha_adj)

    'verification'
    print((g_lss_adj * g_lss).sum() + (b_adj * np.array(lss.bs)).sum() + (g_dil_adj * g_dil).sum())
    alpha = lss.solve()
    print((g_lss_adj * g_lss).sum() + (alpha_adj * alpha).sum() + (g_dil_adj * g_dil).sum())
    grad_lss = (alpha[:,:,np.newaxis] * np.array(G_lss)).sum(1) + np.array(g_lss)
    dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
    grad_dil = dil[:,np.newaxis] * dJ
    print(windowed_mean(grad_lss) + windowed_mean(grad_dil))

