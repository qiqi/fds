import os
import sys
import time
import gzip
import shutil
import tempfile
import argparse
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *
from fds.checkpoint import *

M_MODES = 16             # number of unstable modes
STEPS_PER_SEGMENT = 100  # number of time steps per chunk
K_SEGMENTS = 100         # number of time chunks
STEPS_RUNUP = 0          # additional run up time steps
TIME_PER_STEP = 1E-4
SIMULTANEOUS_RUNS = 3    # max number of simultaneous MPI runs

# modify to point to openfoam binary
REF_WORK_PATH = os.path.join(my_path, '../../pitzdaily/ref')
BASE_PATH = os.path.join(my_path, 'pitzdaily')
HDF5_PATH = os.path.join(BASE_PATH, 'hdf5')
S_BASELINE = 10

H5FOAM = os.path.join(my_path, '../tools/openfoam4/scripts/h5_to_foam.py')
FOAMH5 = os.path.join(my_path, '../tools/openfoam4/scripts/foam_to_h5.py')
PYTHON = '/usr/bin/python'

if not os.path.exists(BASE_PATH):
    os.mkdir(BASE_PATH)
if not os.path.exists(HDF5_PATH):
    os.mkdir(HDF5_PATH)

# modify to point to openfoam binary
pisofoam_bin = '/opt/openfoam4/platforms/linux64GccDPInt32Opt/bin/pisoFoam'

def read_field(data_path):
    field = []
    with gzip.open(os.path.join(data_path, 'U.gz'), 'rb') as f:
        line = f.readline().strip()
        while not line.startswith('internalField'):
            line = f.readline().strip()
        n_lines = int(f.readline().strip())
        assert f.readline().strip() == '('
        for i in range(n_lines):
            line = f.readline().strip()
            assert line.startswith('(') and line.endswith(')')
            field.extend(line[1:-1].split())
        assert f.readline().strip() == ')'
    return array(field, float)

def solve(u0, s, nsteps, run_id, interprocess):
    print('Starting solve, run_id = ', run_id)
    work_path = os.path.join(BASE_PATH, run_id)
    u1 = os.path.join(work_path, 'final.hdf5')
    J_npy = os.path.join(work_path, 'prob.npy')
    if not (os.path.exists(u1) and os.path.exists(J_npy)):
        if os.path.exists(work_path):
            shutil.rmtree(work_path)
        check_call([PYTHON, H5FOAM, REF_WORK_PATH, u0, work_path, '0'])
        controlDict = os.path.join(work_path, 'system/controlDict')
        with open(controlDict, 'rt') as f:
            original = f.read()
        final_time = nsteps * TIME_PER_STEP
        assert 'endTime         1;' in original
        modified = original.replace(
                'endTime         1;',
                'endTime         {0};'.format(final_time))
        with open(controlDict, 'wt') as f:
            f.write(modified)
        for u_file in ['U.gz', 'U_0.gz']:
            u_file = os.path.join(work_path, '0', u_file)
            with gzip.open(u_file, 'rb') as f:
                original = f.read()
            modified = original.replace(
                    'value           uniform (10 0 0);'.encode(),
                    'value           uniform ({0} 0 0);'.format(s).encode())
            with gzip.open(u_file, 'wb') as f:
                f.write(modified)
        with open(os.path.join(work_path, 'out'), 'wt') as f:
            check_call(pisofoam_bin, cwd=work_path, stdout=f, stderr=f)
        check_call([PYTHON, FOAMH5, work_path, str(final_time), u1])
        shutil.rmtree(os.path.join(work_path, '0'))
        shutil.rmtree(os.path.join(work_path, 'system'))
        shutil.rmtree(os.path.join(work_path, 'constant'))
        transient_paths = [os.path.join(work_path, str(i * TIME_PER_STEP))
                           for i in range(1, nsteps+1)]
        J = array([read_field(p) for p in transient_paths])
        for p in transient_paths:
            shutil.rmtree(p)
        save(J_npy, J)
    else:
        J = load(J_npy)
    return u1, J

def getHostDir(run_id):
    return os.path.join(HDF5_PATH, run_id)

if __name__ == '__main__':
    u0 = os.path.join(REF_WORK_PATH, 'final.hdf5')

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
                    epsilon=1E-3,
                    checkpoint_path=BASE_PATH,
                    simultaneous_runs=SIMULTANEOUS_RUNS,
                    get_host_dir=getHostDir)
    else:
        J, G = continue_shadowing(solve,
                                  S_BASELINE,
                                  checkpoint,
                                  K_SEGMENTS,
                                  STEPS_PER_SEGMENT,
                                  epsilon=1E-3,
                                  checkpoint_path=BASE_PATH,
                                  simultaneous_runs=SIMULTANEOUS_RUNS,
                                  get_host_dir=getHostDir)
