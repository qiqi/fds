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

if __name__ == '__main__':
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.mkdir(base_dir)
    
    fname1 = os.path.join(base_dir, 'save1.les')
    shutil.copy(ref_fname, fname1)
    save_compressible_les(fname1, initial_state, verbose=True)
    reloaded_state_1 = load_compressible_les(fname1, verbose=True)

    fname2 = os.path.join(base_dir, 'save2.les')
    shutil.copy(ref_fname, fname2)
    save_compressible_les(fname2, initial_state, verbose=True)
    reloaded_state_2 = load_compressible_les(fname2, verbose=True)

    for k in reloaded_state_1:
        if k != 'STEP':
            if (reloaded_state_1[k] == reloaded_state_2[k]).all():
                print(' matches  ', k)
            else:
                print(' does not match  ', k)
