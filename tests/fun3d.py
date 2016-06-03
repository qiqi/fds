import os
import sys
import time
import shutil
import tempfile
from subprocess import *
from multiprocessing.pool import ThreadPool

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

BASE_PATH = os.path.join(my_path, 'fun3d')
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

AVAILABLE_NODES_FILE = os.path.join(BASE_PATH, 'available_nodes')
if os.path.exists(AVAILABLE_NODES_FILE):
    os.remove(AVAILABLE_NODES_FILE)

# change this to the number of MPI processes for each run
MPI_NP = 2

# change this to the max number of simultaneous MPI runs
# the product of MPI_NP and SIMULTANEOUS_RUNS should be the PBS allocation size
SIMULTANEOUS_RUNS = 24

# change this to the total number of grid points
NUM_CV = 116862

# change this a directory with final.data.* files, so that I know
# how to distribute an initial condition into different ranks
REF_WORK_PATH = os.path.join(
        os.sep,'u','enielsen','runs','chaos','airfoil','runup')
REF_WORK_PATH = '.'

# modify to point to fun3d binary
fun3d_bin = os.path.join(os.sep,'u','enielsen','GIT','Master','fun3d','optimized','FUN3D_90','nodet_mpi')

from fds import finite_difference_shadowing

class grab_from_PBS_NODEFILE:
    '''
    Grab num_procs processes to launch MPI job on a subset of allocation.
    Book keeping is done with a file named 'available_nodes' at BASE_PATH.
    Needs a lock to prevent concurrent IO to that file.
    Remember to call release when MPI job finishes.
    '''
    def __init__(self, num_procs, BASE_PATH, lock):
        self.lock = lock
        self.grab(num_procs)

    def grab(self, num_procs):
        with self.lock:
            if not os.path.exists(AVAILABLE_NODES_FILE):
                available_nodes = open(os.environ['PBS_NODEFILE']).readlines()
                if len(available_nodes) != MPI_NP * SIMULTANEOUS_RUNS:
                    msg = '{0} processees in $PBS_NODEFILE cannot be split' + \
                          'into {1} simultaneous MPI runs of size {2}'
                    raise RuntimeError(msg.format(
                        len(available_nodes), SIMULTANEOUS_RUNS, MPI_NP))
            else:
                available_nodes = open(AVAILABLE_NODES_FILE).readlines()
            if len(available_nodes) < num_procs:
                msg = 'Trying to grab {0} processes from {1}'
                raise ValueError(msg.format(num_procs, len(available_nodes)))
            self.grabbed_nodes = available_nodes[:num_procs]
            self.available_nodes = available_nodes[num_procs:]
            with open(AVAILABLE_NODES_FILE, 'wt') as f:
                f.writelines(available_nodes)

    def release(self):
        with self.lock:
            available_nodes = open(AVAILABLE_NODES_FILE).readlines()
            available_nodes += self.grabbed_nodes
            with open(AVAILABLE_NODES_FILE, 'wt') as f:
                f.writelines(available_nodes)

    def write_to_sub_nodefile(self, filename):
        with open(filename, 'wt') as f:
            f.writelines(self.grabbed_nodes)

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

def solve(u0, mach, nsteps, run_id, lock):
    work_path = os.path.join(BASE_PATH, run_id)
    initial_data_files = [os.path.join(work_path, 'initial.data.'+ str(i))
                          for i in range(MPI_NP)]
    final_data_files = [os.path.join(work_path, 'final.data.'+ str(i))
                        for i in range(MPI_NP)]
    lift_drag_file = os.path.join(work_path, 'lift_drag.txt')
    if not all([os.path.exists(f) for f in final_data_files]):
        if not os.path.exists(work_path):
            os.mkdir(work_path)
        sub_nodes = grab_from_PBS_NODEFILE(MPI_NP, BASE_PATH, lock)
        sub_nodefile = os.path.join(work_path, 'PBS_NODEFILE')
        sub_nodes.write_to_sub_nodefile(sub_nodefile)
        env = dict(os.environ)
        env['PBS_NODEFILE'] = sub_nodefile
        shutil.copy(os.path.join(my_path,'fun3d.nml'),work_path)
        shutil.copy(os.path.join(my_path,'rotated.b8.ugrid'),work_path)
        shutil.copy(os.path.join(my_path,'rotated.mapbc'),work_path)
        for file_i, u_i in zip(initial_data_files, distribute_data(u0)):
            with open(file_i, 'wb') as f:
                f.write(asarray(u_i, dtype='>d').tobytes())
        outfile = os.path.join(work_path, 'flow.output')
        with open(outfile, 'w', 0) as f:
            Popen(['mpiexec', fun3d_bin,
                   '--write_final_field', '--read_initial_field',
                   '--ncyc', str(nsteps)
                  ], cwd=work_path, env=env, stdout=f, stderr=f)
        savetxt(lift_drag_file, lift_drag_from_text(open(outfile).read()))
        sub_nodes.release()
    J = loadtxt(lift_drag_file)
    u1 = hstack([frombuffer(open(f, 'rb').read(), dtype='>d')
                 for f in final_data_files])
    print len(J), nsteps
    assert len(J) == nsteps
    return ravel(u1), J

Ji, Gi = finite_difference_shadowing(
            solve,
            random.rand(NUM_CV * 5), # change 10000 to num of CV
            0.1,                     # nominal xmach parameter
            22,                      # number of unstable modes
            50,                      # number of time chunks
            1000,                    # number of time steps per chunk
            10,                      # run-up time steps
            epsilon=1E-4,
            verbose=True,
            simultaneous_runs=SIMULTANEOUS_RUNS
         )
