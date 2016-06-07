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

SLEEP_SECONDS_FOR_IO = 5

# change this to the number of MPI processes for each FUN3D instance
MPI_NP = 24

# change this to the max number of simultaneous MPI runs
# the product of MPI_NP and SIMULTANEOUS_RUNS should be the PBS allocation size
SIMULTANEOUS_RUNS = 2

# change this to the total number of grid points
NUM_CV = 116862

# change this a directory with final.data.* files, so that I know
# how to distribute an initial condition into different ranks
REF_WORK_PATH = os.path.join(
        os.sep,'nobackupp8','enielsen','NILSS','PythonTesting','run_data')

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
    def __init__(self, num_procs, BASE_PATH, lock_and_dict):
        self.lock, self.dict = lock_and_dict
        self.grab(num_procs)

    def grab(self, num_procs):
        with self.lock:
            if not 'available_nodes' in self.dict:
                available_nodes = open(os.environ['PBS_NODEFILE']).readlines()
                if len(available_nodes) < MPI_NP * SIMULTANEOUS_RUNS:
                    msg = '{0} processees in $PBS_NODEFILE cannot be split' + \
                          'into {1} simultaneous MPI runs of size {2}'
                    raise RuntimeError(msg.format(
                        len(available_nodes), SIMULTANEOUS_RUNS, MPI_NP))
            else:
                available_nodes = self.dict['available_nodes']
            if len(available_nodes) < num_procs:
                msg = 'Trying to grab {0} processes from {1}'
                raise ValueError(msg.format(num_procs, len(available_nodes)))
            self.grabbed_nodes = available_nodes[:num_procs]
            self.available_nodes = available_nodes[num_procs:]
            self.dict['available_nodes'] = available_nodes

    def release(self):
        with self.lock:
            self.dict['available_nodes'] += self.grabbed_nodes

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
        shutil.copy(os.path.join(REF_WORK_PATH,'fun3d.nml'),work_path)
        shutil.copy(os.path.join(REF_WORK_PATH,'rotated.b8.ugrid'),work_path)
        shutil.copy(os.path.join(REF_WORK_PATH,'rotated.mapbc'),work_path)
        for file_i, u_i in zip(initial_data_files, distribute_data(u0)):
            with open(file_i, 'wb') as f:
                f.write(asarray(u_i, dtype='>d').tobytes())
        outfile = os.path.join(work_path, 'flow.output')
        with open(outfile, 'w', 0) as f:
            Popen(['mpiexec', fun3d_bin,
                   '--write_final_field', '--read_initial_field',
                   '--ncyc', str(nsteps)
                  ], cwd=work_path, env=env, stdout=f, stderr=f).wait()
            time.sleep(SLEEP_SECONDS_FOR_IO)
        savetxt(lift_drag_file, lift_drag_from_text(open(outfile).read()))
        sub_nodes.release()
    J = loadtxt(lift_drag_file).reshape([-1,2])
    u1 = hstack([frombuffer(open(f, 'rb').read(), dtype='>d')
                 for f in final_data_files])
    print 'len(J) = ', len(J), 'nsteps = ', nsteps
    assert len(J) == nsteps
    return ravel(u1), J

# read data after run up
initial_data_files = [os.path.join(REF_WORK_PATH, 'final.data.'+ str(i))
                    for i in range(MPI_NP)]
u0 = hstack([frombuffer(open(f, 'rb').read(), dtype='>d')
             for f in initial_data_files])

Ji, Gi = finite_difference_shadowing(
            solve,
            u0,                      # 5 variables per CV
            0.1,                     # nominal xmach parameter
            22,                      # number of unstable modes
            50,                      # number of time chunks
            1000,                    # number of time steps per chunk
            0,                       # needs no more run up, since we assume
                                     # it is already down
            epsilon=1E-4,
            verbose=True,
            simultaneous_runs=SIMULTANEOUS_RUNS
         )
