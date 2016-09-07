from __future__ import print_function

import os
import sys
import time
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(my_path, '..'))

from fds import *
from fds.checkpoint import *
from fds.states import PrimalState

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

def files(state):
    for i in range(MPI_NP):
        yield '.'.join([state.name, str(i)])

def mpi_read(mpi_comm, state):
    fname = '.'.join([state.name, str(mpi_comm.rank)])
    return frombuffer(open(fname, 'rb').read(), dtype='>d')

def mpi_write(mpi_comm, state, raw_data):
    fname = '.'.join(state.name, str(mpi_comm.rank))
    with open(fname, 'wb') as f:
        f.write(asarray(raw_data, dtype='>d').tobytes())

def lift_drag_from_text(text):
    lift_drag = []
    for line in text.split('\n'):
        line = line.strip().split()
        if len(line) == 4 and line[0] == 'Lift' and line[2] == 'Drag':
            lift_drag.append([line[1], line[3]])
    return array(lift_drag, float)

def solve(u0, mach, nsteps, run_id, interprocess):
    print('Starting solve, mach, nsteps, run_id = ', mach, nsteps, run_id)
    work_path = os.path.join(BASE_PATH, run_id)
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
        for file_i, u_i in zip(initial_data_files, files(u0)):
            shutil.copy(u_i, file_i)
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
    u1 = PrimalState(os.path.join(work_path, 'final.data'))
    assert len(J) == nsteps
    return u1, J

if __name__ == '__main__':
#def test_fun3d():
    u0 = PrimalState(os.path.join(REF_WORK_PATH, 'final.data'),
                     mpi_run_cmd=['mpiexec', '-np', str(MPI_NP)],
                     mpi_read=mpi_read, mpi_write=mpi_write)
    shadowing(solve,
              u0,
              XMACH,
              M_MODES,
              2,
              STEPS_PER_SEGMENT,
              STEPS_RUNUP,
              epsilon=1E-4,
              checkpoint_path=BASE_PATH,
              simultaneous_runs=SIMULTANEOUS_RUNS)

    checkpoint = load_last_checkpoint(BASE_PATH, M_MODES)
    J, G = continue_shadowing(solve,
                              XMACH,
                              checkpoint,
                              K_SEGMENTS,
                              STEPS_PER_SEGMENT,
                              epsilon=1E-4,
                              checkpoint_path=BASE_PATH,
                              simultaneous_runs=SIMULTANEOUS_RUNS)

    assert 1 < J[0] < 4
    assert 10 < J[1] < 40
    assert -0.1 < G[0] < 0.5
    assert 1 < G[1] < 8
