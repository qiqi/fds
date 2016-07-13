import os
import sys
import shutil
import tempfile
from subprocess import *

import matplotlib
matplotlib.use('Agg')
from pylab import *
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *
from fds.checkpoint import *
import dowell


def solve(u, s, nsteps, run_id=None, lock=None):
    tmp_path = tempfile.mkdtemp()
    
    n_grid = u.shape[0] # number of grid points
    dt = 0.001 #time step

    out = u.copy()
    J = zeros(nsteps)
    s_arr = array([s,150.0,0.3,0.1,0.0])
    dowell.c_run_primal(out, s_arr, J, nsteps, n_grid, dt)
    shutil.rmtree(tmp_path)
    return out, J


