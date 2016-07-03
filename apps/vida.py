import os
import sys
import time
import shutil
import string
import tempfile
import argparse
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *
from fds.checkpoint import *
from fds.cti_restart_io import *

INLET_U = 0.053826972     # nominal inlet velocity
M_MODES = 4               # number of unstable modes
K_SEGMENTS = 100          # number of time chunks
STEPS_PER_SEGMENT = 500   # number of time steps per chunk
STEPS_RUNUP = 10          # additional run up time steps
SLEEP_SECONDS_FOR_IO = 18 # how long to wait for file IO to sync
MPI_NP = 576              # number of MPI processes for each instance
# MPI_NP = 48               # number of MPI processes for each instance
SIMULTANEOUS_RUNS = 3     # max number of simultaneous MPI runs

STATE_VARS = ['bullet_nose:RHOU_BC', 'RHOUM', 'PHIM', 'RHOUM0',
 'bullet_base:RHOU_BC', 'P', 'U', 'outlet:RHOU_BC', 'out_wall:RHOU_BC',
 'inlet:RHOU_BC', 'bullet_wall:RHOU_BC', 'PHIM0']
DUMMY_VARS = ['RHO', 'RHO0', 'T', 'MU_SGS', 'MU_LAM']

# change this a directory with the "box" executable and params.template
REF_WORK_PATH = os.path.abspath(os.path.join(os.environ['HOME'], 'BulletBody-ref'))
PARAMS_TEMPLATE = string.Template(
        open(os.path.join(REF_WORK_PATH, 'vida.template')).read())
STATS_FILES = open(os.path.join(REF_WORK_PATH, 'probe_files')).read().strip().split(' ')
REF_DATA_FILE = os.path.join(REF_WORK_PATH, 'initial.les')
REF_STATE = load_les(REF_DATA_FILE, verbose=False)

BASE_PATH = os.path.join(my_path, 'vida')
S_BASELINE = INLET_U
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

vida_bin = os.path.join(REF_WORK_PATH, 'vida.exe')

def make_data(u):
    if u.ndim != 1:
        raise ValueError('make_data u.ndim = {0} != 1'.format(u.ndim))
    data, start = {}, 0
    for name in STATE_VARS:
        size = REF_STATE[name].size
        data[name] = u[start:start+size]
        start += size
    for name in DUMMY_VARS:
        data[name] = NO_CHANGE
    data['STEP'] = 1
    if start != u.size:
        raise ValueError('make_data u.size = {0} != {2}'.format(u.size, start))
    return data

def solve(u0, inlet_u, nsteps, run_id, interprocess):
    print('Starting solve, inlet_u, nsteps, run_id = ',
                           inlet_u, nsteps, run_id)
    work_path = os.path.join(BASE_PATH, run_id)
    initial_data_file = os.path.join(work_path, 'initial.les')
    final_data_file = os.path.join(work_path, 'result.les')
    stats_files = [os.path.join(work_path, 'PROBES', fname)
                   for fname in STATS_FILES]
    if not os.path.exists(final_data_file) or \
            not all([os.path.exists(f) for f in stats_files]):
        if not os.path.exists(work_path):
            os.mkdir(work_path)
            for subdir in 'STATS ISOS PROBES ZONES LOGS MONITOR SOLUT CUTS'.split():
                os.mkdir(os.path.join(work_path, subdir))
        sub_nodes = pbs.grab_from_PBS_NODEFILE(MPI_NP, interprocess)
        sub_nodefile = os.path.join(work_path, 'PBS_NODEFILE')
        sub_nodes.write_to_sub_nodefile(sub_nodefile)
        env = dict(os.environ)
        env['PBS_NODEFILE'] = sub_nodefile
        with open(os.path.join(work_path, 'vida.in'), 'w') as f:
            f.write(PARAMS_TEMPLATE.substitute(
                INLET_U=str(inlet_u), NSTEPS=nsteps+1))
        shutil.copy(REF_DATA_FILE, initial_data_file)
        save_les(initial_data_file, make_data(u0), verbose=False)
        outfile = os.path.join(work_path, 'vida.output')
        with open(outfile, 'w', 8) as f:
            Popen(['mpiexec', '-n', str(MPI_NP), vida_bin],
                  cwd=work_path, env=env, stdout=f, stderr=f).wait()
            time.sleep(SLEEP_SECONDS_FOR_IO)
        sub_nodes.release()
    J = hstack([loadtxt(f) for f in stats_files])
    J = J.reshape([nsteps, -1, 4])[:,:,3]
    solution = load_les(final_data_file, verbose=False)
    u1 = hstack([ravel(solution[state]) for state in STATE_VARS])
    return ravel(u1), J

if __name__ == '__main__':
    u0 = hstack([ravel(REF_STATE[state]) for state in STATE_VARS])
    # solve(u0, INLET_U, 100, 'test_run', None)
    # stop

    checkpoint = load_last_checkpoint(BASE_PATH, M_MODES)
    if checkpoint is None:
        J, G = shadowing(
                    solve,
                    u0,
                    S_BASELINE,
                    M_MODES,
                    K_SEGMENTS,
                    STEPS_PER_SEGMENT,
                    STEPS_RUNUP,
                    epsilon=1E-4,
                    checkpoint_path=BASE_PATH,
                    simultaneous_runs=SIMULTANEOUS_RUNS
                 )
    else:
        J, G = continue_shadowing(solve,
                                  S_BASELINE,
                                  checkpoint,
                                  K_SEGMENTS,
                                  STEPS_PER_SEGMENT,
                                  epsilon=1E-4,
                                  checkpoint_path=BASE_PATH,
                                  simultaneous_runs=SIMULTANEOUS_RUNS)
