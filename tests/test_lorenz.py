import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *

solver_path = os.path.join(my_path, 'solvers', 'lorenz')
solver = os.path.join(solver_path, 'solver')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

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
    J = transpose([J, 100 * ones(J.size)])
    return out, J

#if __name__ == '__main__':
def test_gradient():
    s = linspace(28, 33, 6)
    J, G = zeros([s.size, 2]), zeros([s.size, 2])
    for i, si in enumerate(s):
        print(i)
        Ji, Gi = shadowing(solve, u0, si-28, 2, 10, 1000, 5000)
        J[i,:] = Ji
        G[i,:] = Gi
    assert all(abs(J[:,1] - 100) < 1E-12)
    assert all(abs(G[:,1]) < 1E-12)
    assert all(abs(J[:,0] - ((s-31)**2 + 85)) < 20)
    assert all(abs(G[:,0] - (2 * (s-31))) < 2)

#if __name__ == '__main__':
def test_lyapunov():
    cp_path = os.path.join(my_path, 'lorenz_lyapunov')
    if os.path.exists(cp_path):
        shutil.rmtree(cp_path)
    os.mkdir(cp_path)
    m = 2
    J, G = shadowing(solve, u0, 0, m, 20, 1000, 5000, checkpoint_path=cp_path)
    cp = checkpoint.load_last_checkpoint(cp_path, m)
    L = cp.lss.lyapunov_exponents()

    def exp_mean(x):
        n = len(x)
        w = 1 - exp(range(1,n+1) / sqrt(n))
        x = array(x)
        w = w.reshape([-1] + [1] * (x.ndim - 1))
        return (x * w).sum(0) / w.sum()

    lam1, lam2 = exp_mean(L[:,:5])
    assert 0.5 < lam1 < 1.5
    assert -15 < lam2 < -5
