from __future__ import print_function

import os
import sys
import time
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))

BASE_PATH = os.path.join(my_path, 'fun3d')
assert os.path.exists(BASE_PATH)

sys.path.append(os.path.join(my_path, '..'))
from fds import finite_difference_shadowing

sys.path.append(my_path)
from fun3d import *

class DidNotFinishRunningFUN3D(Exception): pass

def solve_no_run_fun3d(u0, mach, nsteps, run_id, lock):
    work_path = os.path.join(BASE_PATH, run_id)
    initial_data_files = [os.path.join(work_path, 'initial.data.'+ str(i))
                          for i in range(MPI_NP)]
    final_data_files = [os.path.join(work_path, 'final.data.'+ str(i))
                        for i in range(MPI_NP)]
    lift_drag_file = os.path.join(work_path, 'lift_drag.txt')
    if not all([os.path.exists(f) for f in final_data_files]):
        raise DidNotFinishRunningFUN3D
    J = loadtxt(lift_drag_file).reshape([-1,2])
    u1 = hstack([frombuffer(open(f, 'rb').read(), dtype='>d')
                 for f in final_data_files])
    assert len(J) == nsteps
    return ravel(u1), J

try:
    Ji, Gi = finite_difference_shadowing(
                solve_no_run_fun3d,
                u0,
                XMACH,
                M_MODES,
                K_SEGMENTS,
                STEPS_PER_SEGMENT,
                STEPS_RUNUP,
                epsilon=1E-4,
                verbose='save_data'
         )
except DidNotFinishRunningFUN3D:
    pass  # it's fine, just use what we have so far

data = load('lss.npz')
R = data['R']
i = arange(R.shape[1])
print('Lyapunov exponents:')
for i, lyapunov in enumerate(log(abs(R[:,i,i])).mean(0)):
    print(i, lyapunov)
