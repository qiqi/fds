from __future__ import print_function

import h5py
import os
import sys
import time
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *
from fds.checkpoint import *

XMACH = 0.1              # nominal xmach parameter
M_MODES = 16             # number of unstable modes
K_SEGMENTS = 5           # number of time chunks
STEPS_PER_SEGMENT = 100  # number of time steps per chunk
STEPS_RUNUP = 2000       # additional run up time steps
SLEEP_SECONDS_FOR_IO = 1 # how long to wait for file IO to sync
MPI_NP = 4               # number of MPI processes for each FUN3D instance
SIMULTANEOUS_RUNS = 2    # max number of simultaneous MPI runs

# change this a directory with final.data.* files, so that I know
# how to distribute an initial condition into different ranks
REF_WORK_PATH = os.path.join(my_path, 'solvers', 'mock_fun3d')

BASE_PATH = os.path.join(my_path, 'fun3d')
if os.path.exists(BASE_PATH):
    shutil.rmtree(BASE_PATH)
os.mkdir(BASE_PATH)

# modify to point to fun3d binary
fun3d_bin = os.path.join(REF_WORK_PATH, 'fun3d')

if 'PBS_NODEFILE' not in os.environ:
    os.environ['PBS_NODEFILE'] = os.path.join(REF_WORK_PATH, 'PBS_NODEFILE')

def save_hdf5(arr, path):
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('field', data=arr)
        return
def load_hdf5(path):
    with h5py.File(path, 'r') as handle:
        field = handle['/field'][:].copy()
    return field

def get_host_dir(run_id):
    return os.path.join(BASE_PATH, run_id)

def spawn_compute_job(exe, args):
    return call(['mpirun', '-np', '2', 'python', exe] + args)

def distribute_data(u):
    if not hasattr(distribute_data, 'doubles_for_each_rank'):
        distribute_data.doubles_for_each_rank = []
        for i in range(MPI_NP):
            final_data_file = os.path.join(REF_WORK_PATH, 'final.data.'+ str(i))
            with open(final_data_file, 'rb') as f:
                ui = frombuffer(f.read(), dtype='>d')
                distribute_data.doubles_for_each_rank.append(ui.size)
    u_distributed = []
    for n in distribute_data.doubles_for_each_rank:
        assert u.size >= n
        u_distributed.append(u[:n])
        u = u[n:]
    assert u.size == 0
    return u_distributed

def lift_drag_from_text(text):
    lift_drag = []
    for line in text.split('\n'):
        line = line.strip().split()
        if len(line) == 4 and line[0] == 'Lift' and line[2] == 'Drag':
            lift_drag.append([line[1], line[3]])
    return array(lift_drag, float)

def solve(u0, mach, nsteps, run_id, interprocess):
    print('Starting solve, mach, nsteps, run_id = ', mach, nsteps, run_id)
    u0 = load_hdf5(u0)
    work_path = get_host_dir(run_id)
    initial_data_files = [os.path.join(work_path, 'initial.data.'+ str(i))
                          for i in range(MPI_NP)]
    final_data_files = [os.path.join(work_path, 'final.data.'+ str(i))
                        for i in range(MPI_NP)]
    lift_drag_file = os.path.join(work_path, 'lift_drag.txt')
    if not all([os.path.exists(f) for f in final_data_files]) or \
            not os.path.exists(lift_drag_file):
        if not os.path.exists(work_path):
            os.mkdir(work_path)
        sub_nodes = pbs.grab_from_PBS_NODEFILE(MPI_NP, interprocess, True)
        sub_nodefile = os.path.join(work_path, 'PBS_NODEFILE')
        sub_nodes.write_to_sub_nodefile(sub_nodefile)
        env = dict(os.environ)
        env['PBS_NODEFILE'] = sub_nodefile
        shutil.copy(os.path.join(REF_WORK_PATH,'fun3d.nml'),work_path)
        shutil.copy(os.path.join(REF_WORK_PATH,'rotated.b8.ugrid'),work_path)
        shutil.copy(os.path.join(REF_WORK_PATH,'rotated.mapbc'),work_path)
        for file_i, u_i in zip(initial_data_files, distribute_data(u0)):
            with open(file_i, 'wb') as f:
                f.write(asarray(u_i, dtype='>d').tobytes())
        outfile = os.path.join(work_path, 'flow.output')
        with open(outfile, 'w', 8) as f:
            Popen(['mpiexec', '-np', str(MPI_NP), fun3d_bin,
                   '--write_final_field', '--read_initial_field',
                   '--ncyc', str(nsteps), '--xmach', str(mach)
                  ], cwd=work_path, env=env, stdout=f, stderr=f).wait()
            time.sleep(SLEEP_SECONDS_FOR_IO)
        savetxt(lift_drag_file, lift_drag_from_text(open(outfile).read()))
        sub_nodes.release()
    J = loadtxt(lift_drag_file).reshape([-1,2])
    u1 = hstack([frombuffer(open(f, 'rb').read(), dtype='>d')
                 for f in final_data_files])
    assert len(J) == nsteps
    u1 = ravel(u1)
    path = os.path.join(work_path, 'output.h5')
    save_hdf5(u1, path)
    return path, J

def test_fun3d():
    initial_data_files = [os.path.join(REF_WORK_PATH, 'final.data.'+ str(i))
                        for i in range(MPI_NP)]
    u0 = hstack([frombuffer(open(f, 'rb').read(), dtype='>d')
                 for f in initial_data_files])
    path = os.path.join(BASE_PATH, 'u0.h5')
    save_hdf5(u0, path)
    u0 = path
    shadowing(solve,
              u0,
              XMACH,
              M_MODES,
              2,
              STEPS_PER_SEGMENT,
              STEPS_RUNUP,
              epsilon=1E-4,
              checkpoint_path=BASE_PATH,
              simultaneous_runs=SIMULTANEOUS_RUNS,
              get_host_dir=get_host_dir,
              spawn_compute_job=spawn_compute_job)

    checkpoint = load_last_checkpoint(BASE_PATH, M_MODES)
    J, G = continue_shadowing(solve,
                              XMACH,
                              checkpoint,
                              K_SEGMENTS,
                              STEPS_PER_SEGMENT,
                              epsilon=1E-4,
                              checkpoint_path=BASE_PATH,
                              simultaneous_runs=SIMULTANEOUS_RUNS,
                              get_host_dir=get_host_dir,
                              spawn_compute_job=spawn_compute_job)

    assert 1 < J[0] < 4
    assert 10 < J[1] < 40
    assert -0.1 < G[0] < 0.5
    assert 1 < G[1] < 8

if __name__ == '__main__':
    test_fun3d()
