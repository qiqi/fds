import os
import sys
import time
import shutil
import tempfile
from subprocess import *
from multiprocessing.pool import ThreadPool

import matplotlib
#matplotlib.use('Agg')
from pylab import *
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import finite_difference_shadowing

def lift_drag_from_text(text):
    lift_drag = []
    for line in text:
        line = line.strip().split()
        if len(line) == 4 and line[0] == 'Lift' and line[2] == 'Drag':
            lift_drag.append([line[1], line[3]])
    return array(lift_drag, float)

def solve(u0, mach, nsteps, run_id):
    base_path = os.path.join(my_path, 'fun3d')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    work_path = os.path.join(base_path, run_id)
    initial_data_file = os.path.join(work_path, 'initial.data.0')
    final_data_file = os.path.join(work_path, 'final.data.0')
    lift_drag_file = os.path.join(work_path, 'lift_drag.txt'))
    if not os.path.exists(final_data_file):
        if not os.path.exists(work_path):
            os.mkdir(work_path)
        with open(initial_data_file, 'wb') as f:
            f.write(asarray(u0, dtype='>d').tobytes())
        screen_output = check_output([
            fun3d_bin, '--read_initial_field', '--write_final_field',
            '--xmach', str(mach), '--ncyc', str(nsteps)
        ], cwd=work_path)
        savetxt(lift_drag_file, lift_drag_from_text(screen_output))
    J = loadtxt(lift_drag_file)
    assert len(J) == nsteps
    with open(final_data_file, 'rb') as f:
        u1 = frombuffer(f.read(), dtype='>d')
    return ravel(u1), J

# modify to point to fun3d binary
fun3d_bin = os.path.join(my_path, '..', '..', 'fun3d', 'fun3d')

Ji, Gi = finite_difference_shadowing(
            solve,
            random.rand(10000 * 5), # change 10000 to num of CV
            0.25,                   # nominal xmach parameter
            30,                     # number of unstable modes
            50,                     # number of time chunks
            1000,                   # number of time steps per chunk
            1000,                   # run-up time steps
            epsilon=1E-4,
            verbose=True
         )
