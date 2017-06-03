from __future__ import division
import os
import sys
import time
import shutil
import string
import tempfile
import argparse
import subprocess
from pdb import set_trace
from numpy import *
from copy import deepcopy

sys.path.append("/scratch/niangxiu/fds")
from fds import *
from fds.checkpoint import *
from fds.cti_restart_io import *
sys.setrecursionlimit(12000)

def load_compressible_les(fname, verbose=True):
    les = load_les(fname, verbose)
    les['RHO']  = les['RHO']  / 1.18
    les['RHOU'] = les['RHOU'] / 30.0
    les['RHOE'] = les['RHOE'] / 2.5e5
    les['right:RHO_BC'] = les['right:RHO_BC'] / 1.18
    les['right:P_BC']   = les['right:P_BC']   / 1.0e5
    les['right:U_BC']   = les['right:U_BC']   / 30.0
    return les

def save_compressible_les(fname, les0, verbose=True):
    les = deepcopy(les0)
    les['RHO']  = les['RHO']  * 1.18
    les['RHOU'] = les['RHOU'] * 30.0
    les['RHOE'] = les['RHOE'] * 2.5e5
    les['right:RHO_BC'] = les['right:RHO_BC'] * 1.18
    les['right:P_BC']   = les['right:P_BC']   * 1.0e5
    les['right:U_BC']   = les['right:U_BC']   * 30.0
    save_les(fname, les, verbose)

INLET_U = 33.0                  # nominal inlet velocity
M_MODES = 40                    # number of unstable modes
K_SEGMENTS = 400                # number of time chunks
STEPS_PER_SEGMENT = 200         # number of time steps per chunk
STEPS_RUNUP = 200               # additional run up time steps
SLEEP_SECONDS_FOR_IO = 0        # how long to wait for file IO to sync
MPI_NP = 16                     # number of MPI processes for each instance
SIMULTANEOUS_RUNS = 4           # max number of simultaneous MPI runs

STATE_VARS =  [ 'RHO', 'RHOU', 'RHOE',
                'right:RHO_BC', 'right:P_BC', 'right:U_BC' ]
DUMMY_VARS =  [ 'STEP', 'DT', 'TIME', 'MU_LAM', 'CP', 
                'K_LAM', 'MU_SGS', 'K_SGS', 'T', 'P', 'U',
                'cylinder:RHO_BC', 'cylinder:P_BC', 'cylinder:U_BC', 
                'left:RHO_BC', 'left:P_BC', 'left:U_BC'] 

my_path = os.path.abspath('/scratch/niangxiu/fds/apps')
REF_WORK_PATH = os.path.abspath('/scratch/niangxiu/cylinder_fds_ref')
PARAMS_TEMPLATE = string.Template(
        open(os.path.join(REF_WORK_PATH, 'charles.template')).read())
STATS_FILES = ['J_history.txt'] #where to look for quantities of interest
REF_DATA_FILE = os.path.join(REF_WORK_PATH, 'initial.les')
REF_SUPP_FILE = os.path.join(REF_WORK_PATH, 'write_J.py')
REF_STATE = load_compressible_les(REF_DATA_FILE, verbose=False)

BASE_PATH = os.path.join(my_path, 'charles')
S_BASELINE = INLET_U
if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)

charles_bin = os.path.join(REF_WORK_PATH, 'charles.exe')

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
    data['STEP'] = 0
    if start != u.size:
        raise ValueError('make_data u.size = {0} != {2}'.format(u.size, start))
    return data

def solve(u0, inlet_u, nsteps, run_id, interprocess):
    print('Starting solve, inlet_u, nsteps, run_id = ',
                           inlet_u, nsteps, run_id)
    sys.stdout.flush()
    work_path = os.path.join(BASE_PATH, run_id)
    initial_data_file = os.path.join(work_path, 'initial.les')
    final_data_file = os.path.join(work_path, 'final.les')
    stats_files = [os.path.join(work_path,  fname)
                   for fname in STATS_FILES]
    while not os.path.exists(final_data_file) or \
            not all([os.path.exists(f) for f in stats_files]):
        if os.path.exists(work_path):
            shutil.rmtree(work_path)
        os.mkdir(work_path)
        print('Solving in ' + work_path)
        sys.stdout.flush()
        for subdir in 'SOLUT_1 SOLUT_2 SOLUT_3 SOLUT_4'.split():
            os.mkdir(os.path.join(work_path, subdir))
        
        with open(os.path.join(work_path, 'charles.in'), 'w') as f:
            f.write(PARAMS_TEMPLATE.substitute(
                INLET_U=str(inlet_u), NSTEPS=nsteps))
        shutil.copy(REF_DATA_FILE, initial_data_file)
        shutil.copy(REF_SUPP_FILE, work_path)
        save_compressible_les(initial_data_file, make_data(u0), verbose=False)
        outfile = os.path.join(work_path, 'out.output')

        nodes = slurm.grab_from_SLURM_NODELIST(1, interprocess)
        with open(outfile, 'w', 8) as f:
            f.write('{0}'.format(nodes))
            subprocess.call(['mpirun', '--host', ','.join(nodes.grabbed_nodes)
                           , '-n', str(MPI_NP), charles_bin] 
                           , cwd=work_path, stdout = f, stderr=f)
            subprocess.call(['mpirun', '--host', ','.join(nodes.grabbed_nodes)
                           , 'python3', 'write_J.py'] 
                           , cwd=work_path, stdout = f, stderr=f)
            # Popen(['mpiexec', '-n', str(MPI_NP), charles_bin],
                  # cwd=work_path, stdout=f, stderr=f).wait()
            # Popen(['python', 'write_J.py'],
                  # cwd=work_path, stdout=f, stderr=f).wait()
        time.sleep(SLEEP_SECONDS_FOR_IO)
        nodes.release()

        if not os.path.exists(final_data_file) or \
                not all([os.path.exists(f) for f in stats_files]):
            failed_path = work_path + '_failed'
            if os.path.exists(failed_path):
                shutil.rmtree(failed_path)
            shutil.move(work_path, failed_path)
    J = hstack([loadtxt(f) for f in stats_files])
    J = J.T.reshape([nsteps, 4]) # change to output a nsteps by x array where x is the # of QoIs
    print(J.shape)
    assert J.shape == (nsteps, 4)
    solution = load_compressible_les(final_data_file, verbose=False)
    u1 = hstack([ravel(solution[state]) for state in STATE_VARS])
    os.remove(work_path + '/initial.les')
    os.remove(work_path + '/final.les')
    return ravel(u1), J

if __name__ == '__main__':
    u0 = hstack([ravel(REF_STATE[state]) for state in STATE_VARS])

    checkpoint = load_last_checkpoint(BASE_PATH, M_MODES)
    # fds.test_linearity(solve, S_BASELINE, checkpoint, STEPS_PER_SEGMENT, epsilon=1e-4, simultaneous_runs=SIMULTANEOUS_RUNS)

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
