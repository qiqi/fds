from __future__ import division

import os
import sys
import shutil
import tempfile
from subprocess import *

import numpy as np
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds.timeseries import windowed_mean_weights, windowed_mean
from fds.checkpoint import load_checkpoint
from fds.segment import run_segment, trapez_mean, adjoint_segment
from fds.fds import AdjointWrapper, RunWrapper
from fds.timedilation import TimeDilation, TimeDilationExact
from fds.state import state_dot
from fds import *

solver_path = os.path.join(my_path, 'solvers', 'rijke')
solver = os.path.join(solver_path, 'solver')
adj_solver = os.path.join(solver_path, 'adjoint')
tan_solver = os.path.join(solver_path, 'tangent')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

def solve(u, s, nsteps):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray([10.0, 10.0, 28.0, 8./3, 0.01, \
        0.3, s, 0.05, 0.01, 0.04], dtype='>d').tobytes())
    shutil.copyfile(os.path.join(solver_path,'Dcheb.bin'), \
    os.path.join(tmp_path,'Dcheb.bin'))
    call([solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'objective.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, J

def tangent(u, s, v, ds, nsteps):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'tan-input.bin'), 'wb') as f:
        f.write(asarray(v, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray([10.0, 10.0, 28.0, 8./3, 0.01, \
        0.3, s, 0.05, 0.01, 0.04], dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'tan-param.bin'), 'wb') as f:
        f.write(asarray([0., 0., 0., 0., 0., 0., ds, 0., 0., 0.\
        ], dtype='>d').tobytes())
    shutil.copyfile(os.path.join(solver_path,'Dcheb.bin'), \
    os.path.join(tmp_path,'Dcheb.bin'))
    call([tan_solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        u = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'tan-output.bin'), 'rb') as f:
        v = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'J.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'dJ.bin'), 'rb') as f:
        dJ = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return u, v, J, dJ

def adjoint(u, s, nsteps, ua):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray([10.0, 10.0, 28.0, 8./3, 0.01, \
        0.3, s, 0.05, 0.01, 0.04], dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'adj-input.bin'), 'wb') as f:
        f.write(asarray(ua, dtype='>d').tobytes())
    shutil.copyfile(os.path.join(solver_path,'Dcheb.bin'), \
    os.path.join(tmp_path,'Dcheb.bin'))
    call([adj_solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'adj-output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'dJds.bin'), 'rb') as f:
        dJds = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, dJds

def test_rijke_adjoint():
    m = 2
    s = 0.865
    steps_per_segment = 10000
    cp_path = 'tests/rijke_adj'
    if os.path.exists(cp_path):
        shutil.rmtree(cp_path)
    os.mkdir(cp_path)
    J, dJds_tan = shadowing(solve, u0, s, m, 100, steps_per_segment, 100000,
                            checkpoint_path=cp_path, tangent_run=tangent)
    print("Tangent: ",dJds_tan)
    dJds_adj = adjoint_shadowing(solve, adjoint, s, m, cp_path)
    print("Adjoint: ", dJds_adj[6])
    assert abs(dJds_tan[0] - dJds_adj[6]) < 1E-10

if __name__ == '__main__':
#    s = 0.865
#    u0 = random.rand(33)
#    v0 = random.rand(33)
#    w1 = random.rand(33)
#    u1, v1, J, dJ = tangent(u0,s,v0,0,100)
#    print(dot(v1,w1))
#    w0, dJds = adjoint(u0,s,100,w1)
#    print(dot(v0,w0))
#def lorenz_adjoint_oldfashioned_test():
    m = 2
    s = 0.865
    steps_per_segment = 10000
    cp_path = 'tests/lorenz_adj'
    if os.path.exists(cp_path):
        shutil.rmtree(cp_path)
    os.mkdir(cp_path)
    cp = shadowing(solve, u0, s, m, 3, steps_per_segment, 100000,
                   checkpoint_path=cp_path, tangent_run=tangent,
                   return_checkpoint=True)
    u0, _, v, lss, G_lss, g_lss, J, G_dil, g_dil = cp
    g_lss = np.array(g_lss)
    J = np.array(J)
        
    dJ = trapez_mean(J, 1) - J[:,-1]
    print("lss is: ",np.array(g_lss))
    print("Size of lss is:", np.array(g_lss).size, np.array(g_lss).shape)

    assert dJ.ndim == 2 and dJ.shape[1] == 1

    win_lss = windowed_mean_weights(dJ.shape[0])
    g_lss_adj = win_lss[:,newaxis]
    alpha_adj_lss = win_lss[:,newaxis] * np.array(G_lss)[:,:,0]

    win_dil = windowed_mean_weights(dJ.shape[0] - 1)
    dil_adj = win_dil * ravel(dJ)[:-1]
    g_dil_adj = dil_adj / steps_per_segment
    alpha_adj_dil = dil_adj[:,newaxis] * array(G_dil)[1:] / steps_per_segment

    alpha_adj = alpha_adj_lss
    alpha_adj[:-1] += alpha_adj_dil
    b_adj = lss.adjoint(alpha_adj)
    bs = np.array(lss.bs)

    'verification'
    print()
    print((g_lss_adj * g_lss).sum() + (b_adj * bs).sum() + (g_dil_adj * g_dil[1:]).sum())
    alpha = lss.solve()
    print((g_lss_adj * g_lss).sum() + (alpha_adj * alpha).sum() + (g_dil_adj * g_dil[1:]).sum())
    grad_lss = (alpha[:,:,np.newaxis] * np.array(G_lss)).sum(1) + g_lss
    dil = ((alpha[:-1] * array(G_dil)[1:]).sum(1) + np.array(g_dil)[1:]) / steps_per_segment
    grad_dil = dil[:,np.newaxis] * dJ[:-1]
    print(windowed_mean(grad_lss) + windowed_mean(grad_dil))

    dJds_adj = 0
    w = zeros_like(u0)
    for k in reversed(range(cp.lss.K_segments())):
        #k = cp.lss.K_segments() - 1
        print('k = ', k)

        print((g_lss_adj * g_lss)[:k+1].sum() + (b_adj * bs)[:k+1].sum() + (g_dil_adj * g_dil[1:])[:k].sum() + dJds_adj + \
                state_dot(v, w))

        cp_file = 'm{}_segment{}'.format(m, k)
        u0, V, v, _,_,_,_,_,_ = load_checkpoint(os.path.join(cp_path, cp_file))
        w0, dJds = adjoint_segment(AdjointWrapper(adjoint),
                                  u0, w, s, k, steps_per_segment, g_lss_adj[k])

        time_dil = TimeDilation(RunWrapper(solve), u0, s, 'time_dilation_test', 4)
        V = time_dil.project(V)
        # v0 = time_dil.project(v)

        # _, v1 = lss.checkpoint(V, v0)

        # print((g_lss * g_lss_adj)[:k].sum() + (b_adj * bs)[:k].sum() + (g_dil_adj * g_dil[1:])[:k].sum() + dJds_adj + \
        #       g_lss[k] * g_lss_adj[k] + dot(b_adj[k], bs[k]))
        # print((g_lss * g_lss_adj)[:k].sum() + (b_adj * bs)[:k].sum() + (g_dil_adj * g_dil[1:])[:k].sum() + dJds_adj + \
        #       dJds[3] + dot(v1,w0) + dot(b_adj[k], bs[k]))

        w1 = lss.adjoint_checkpoint(V, w0, b_adj[k])
        w2 = time_dil.project(w1)
        print((g_lss * g_lss_adj)[:k].sum() + (b_adj * bs)[:k].sum() + (g_dil_adj * g_dil[1:])[:k].sum() + dJds_adj + \
              dJds[3] + state_dot(v,w2))

        if k > 0:
            w3 = time_dil.adjoint_contribution(w2, g_dil_adj[k-1])
            print((g_lss * g_lss_adj)[:k].sum() + (b_adj * bs)[:k].sum() + (g_dil_adj * g_dil[1:])[:k-1].sum() + dJds_adj + \
                  dJds[3] + state_dot(v,w3))
            w = w3
        else:
            w = w2
        dJds_adj += dJds[3]

    print('Final:')
    print(dJds_adj)
