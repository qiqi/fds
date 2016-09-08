import os
import sys

from numpy import *
from scipy.integrate import odeint

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

from fds import *

dt = 0.05

def dudt(u, s):
    z0 = 0
    x, y, z, x0, y0 = u
    dxdt = 10 * (y - x)
    dydt = x * (s - z + z0) - y
    dzdt = x * y - 8./3 * (z - z0)
    dx0dt = y0
    dy0dt = (z - x0**2) * y0 - x0 * (2*pi)**2
    return array([dxdt, dydt, dzdt, dx0dt, dy0dt])

def solve(u, s, nsteps):
    x = empty([nsteps, 2])
    for i in range(nsteps):
        u = odeint(lambda u,t : dudt(u, s), u, [0, dt])[1]
        x[i,0] = u[2]
        x[i,1] = u[3]**2
    return u, x

#if __name__ == '__main__':
def test_lyapunov():
    m, steps_per_segment, n_segment = 5, 50, 50
    u = ones(5)
    cp = shadowing(solve, u, 28, m, n_segment, steps_per_segment, 100,
                   run_ddt=0, return_checkpoint=True)
    L = cp.lss.lyapunov_exponents() / (steps_per_segment * dt)
    L, L_err = timeseries.mean_std(L)
    L_min = L - L_err * 3
    L_max = L + L_err * 3
    assert L_min[0] < +0.90 < L_max[0]
    assert L_min[1] < +0.25 < L_max[1]
    assert L_min[2] < +0.00 < L_max[2]
    assert L_min[3] < -4.98 < L_max[3]
    assert L_min[4] < -6.40 < L_max[4]
    v = cp.lss.lyapunov_covariant_vectors()
    assert v.shape == (m, n_segment+1, m)
