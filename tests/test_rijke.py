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
    s = 0.865
    u0 = random.rand(33)
    v0 = random.rand(33)
    w1 = random.rand(33)
    u1, v1, J, dJ = tangent(u0,s,v0,0,100)
    print(dot(v1,w1))
    w0, dJds = adjoint(u0,s,100,w1)
    print(dot(v0,w0))
