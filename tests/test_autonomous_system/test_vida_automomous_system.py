import os
import sys
import shutil
import string
import subprocess

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '../..'))

from fds.cti_restart_io import *

ref_fname = os.path.join(my_path, '..', 'data', 'cti-sample-restart-file.les')
initial_state = load_les(ref_fname, verbose=False)
base_dir = os.path.join(my_path, 'vida')

def run_vida_in(run_dir, state, steps):
    print('Running {0} steps'.format(steps))
    os.mkdir(run_dir)
    template = open(os.path.join(my_path, 'vida.template')).read()
    template = string.Template(template)
    fname = os.path.join(run_dir, 'initial.les')
    shutil.copy(ref_fname, fname)
    state['STEP'] = 1
    save_les(fname, state, verbose=False)
    with open(os.path.join(run_dir, 'vida.in'), 'w') as f:
        f.write(template.substitute(NSTEPS=str(steps+1)))
    with open(os.path.join(run_dir, 'vida.out'), 'w') as f:
        subprocess.check_call('/home/qiqi/BulletBody-ref/vida.exe',
                              cwd=run_dir, stdout=f, stderr=f)
    fname = os.path.join(run_dir, 'result.les')
    return load_les(fname, verbose=False)

if __name__ == '__main__':
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir(base_dir)
    intermediate_state = run_vida_in(os.path.join(base_dir, 'first_5_steps'),
                                     initial_state, 5)
    final_state_1 = run_vida_in(os.path.join(base_dir, 'second_5_steps'),
                                     intermediate_state, 5)
    final_state_2 = run_vida_in(os.path.join(base_dir, 'all_10_steps_at_once'),
                                     initial_state, 10)
    for k in final_state_1:
        if k != 'STEP':
            if (final_state_1[k] == final_state_2[k]).all():
                print(k, ' matches')
            else:
                print(k, ' does not match')
