import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *

solver_path = os.path.join(my_path, 'solvers', 'circular')
solver = os.path.join(solver_path, 'solver')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

BASE_DIR = os.path.join(my_path, 'checkpoint_test')
if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)
os.mkdir(BASE_DIR)

def solve(u, s, nsteps):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray(s, dtype='>d').tobytes())
    call([solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'objective.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, J[:,newaxis]

def test_checkpoint():
    s = 1
    m_modes = 2
    segments = 20
    shadowing(solve, u0, s, m_modes, segments, 100, 0, checkpoint_path=BASE_DIR)
    cp = checkpoint.load_last_checkpoint(BASE_DIR, m_modes)
    assert cp.lss.K_segments() == segments
    assert cp.lss.m_modes() == m_modes
    J, G = continue_shadowing(solve, s, cp, segments + 10, 100)
    assert abs(G - 1) < 0.1
