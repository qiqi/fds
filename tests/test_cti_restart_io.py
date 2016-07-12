import os
import sys
import shutil
import tempfile
from subprocess import *

from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds.cti_restart_io import *

def assert_states(state):
    assert state['bullet_nose:RHOU_BC'].shape == (185, 3)
    assert state['RHOUM'].shape == (2695, 3)
    assert state['MU_LAM'].shape == (2695,)
    assert state['MU_SGS'].shape == (2695,)
    assert state['PHIM'].shape == (2695,)
    assert state['RHO0'].shape == (2695,)
    assert state['RHOUM0'].shape == (2695, 3)
    assert state['bullet_base:RHOU_BC'].shape == (89, 3)
    assert state['P'].shape == (2695,)
    assert state['U'].shape == (2695, 3)
    assert state['T'].shape == (2695,)
    assert state['RHO'].shape == (2695,)
    assert state['outlet:RHOU_BC'].shape == (153, 3)
    assert state['out_wall:RHOU_BC'].shape == (224, 3)
    assert state['inlet:RHOU_BC'].shape == (185, 3)
    assert state['bullet_wall:RHOU_BC'].shape == (80, 3)
    assert state['PHIM0'].shape == (2695,)

#if __name__ == '__main__':
def test_cti_io():
    fname = os.path.join(my_path, 'data', 'cti-sample-restart-file.les')
    state = load_les(fname, verbose=True)
    assert state['STEP'] == 10
    assert_states(state)
    state['STEP'] = 1
    state['something_random'] = 0
    tmp_fname = os.path.join(my_path, 'tmp_data.les')
    assert not os.path.exists(tmp_fname)
    shutil.copy(fname, tmp_fname)
    save_les(tmp_fname, state, verbose=True)
    state = load_les(tmp_fname, verbose=False)
    assert state['STEP'] == 1
    assert_states(state)
    os.remove(tmp_fname)
