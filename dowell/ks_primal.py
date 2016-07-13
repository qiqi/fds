import sys
from numpy import *
import pde


def ks_primal(u0, s, n_steps):
    n_grid = u0.shape[0] # number of grid points
    dt = 0.01 #time step

    u = u0.copy()
    pde.c_run_primal(u, s, n_steps, n_grid, dt)
    return u


s = 0.5
u0 = zeros(127)
u0[64] = 0.1

n_steps = 10000

u = ks_primal(u0,s,n_steps)

for i in range(u0.shape[0]):
    print u[i], u0[i]
