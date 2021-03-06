import os
import sys
import shutil
import string
import subprocess

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '../..'))
sys.path.append(os.path.join(my_path, '../../apps'))

from fds.cti_restart_io import *
from charles import load_compressible_les, save_compressible_les

ref_fname = os.path.join(my_path, '..', 'data', 'charles-sample-restart-file.les')
initial_state = load_compressible_les(ref_fname, verbose=True)
base_dir = os.path.join(my_path, 'charles')

def run_charles_in(run_dir, state, steps):
    print('Running {0} steps'.format(steps))
    os.mkdir(run_dir)
    template = open(os.path.join(my_path, 'charles.template')).read()
    template = string.Template(template)
    fname = os.path.join(run_dir, 'initial.les')
    shutil.copy(ref_fname, fname)
    state['STEP'] = 1
    save_compressible_les(fname, state, verbose=True)
    with open(os.path.join(run_dir, 'charles.in'), 'w') as f:
        f.write(template.substitute(NSTEPS=str(steps+1)))
    with open(os.path.join(run_dir, 'charles.out'), 'w') as f:
        subprocess.check_call('/scratch/niangxiu/cylinder_fds_ref/charles.exe',
                              cwd=run_dir, stdout=f, stderr=f)
    fname = os.path.join(run_dir, 'final.les')
    return load_compressible_les(fname, verbose=True)

DUMMY_VARS =  [ 'STEP', 'DT', 'TIME', 'MU_LAM', 'CP', 
        'K_LAM', 'MU_SGS', 'K_SGS', 'T', 'P', 'U', 
        'left:P_BC', 'left:U_BC', 'left:RHO_BC',
        'cylinder:RHO_BC', 'cylinder:P_BC', 'cylinder:U_BC']
# DUMMY_VARS = []
print(DUMMY_VARS)

if __name__ == '__main__':
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    os.mkdir(base_dir)
    intermediate_state1 = run_charles_in(os.path.join(base_dir, 'first_50_steps'),
                                     initial_state, 50)
    for v in DUMMY_VARS:
        intermediate_state1[v] = NO_CHANGE
    final_state_1 = run_charles_in(os.path.join(base_dir, 'second_50_steps'),
                                     intermediate_state1, 50)

    intermediate_state2 = run_charles_in(os.path.join(base_dir, 'no_stop_first_50_steps'),
                                     initial_state, 50)
    final_state_2 = run_charles_in(os.path.join(base_dir, 'no_stop_second_50_steps'),
                                     intermediate_state2, 50)
    # final_state_2 = run_charles_in(os.path.join(base_dir, 'all_100_steps_at_once'),
                                     # initial_state, 100)
    for k in final_state_1:
        if k != 'STEP' and k != 'TIME':
            if (final_state_1[k] == final_state_2[k]).all():
                print(' matches', k)
            else:
                print(' does not match', k)
