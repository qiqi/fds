# This file reads the checkpoint files, computes CLVs, call charles.exe for 0 step, 
# and genereate the flow field solution for state variables: rho, rhoE, rhoU

from __future__ import division
import os
import sys
import time
import shutil
import string
import tempfile
import argparse
import subprocess
from multiprocessing import Manager
from numpy import *
import pickle
from charles import solve, save_compressible_les, make_data

sys.path.append("/scratch/niangxiu/fds")
from fds import *
from fds.checkpoint import *
from fds.cti_restart_io import *
from fds.compute import run_compute
from fds.timedilation import TimeDilation
sys.setrecursionlimit(12000)

total_MODES = 30
K_SEGMENTS = arange(250, 351)
# K_SEGMENTS = (300,)
MPI_NP = 2
INLET_U_BASE = 33 
norm_arr = zeros(len(K_SEGMENTS))
m_cells = 7.5e5

MY_PATH = os.path.abspath('/scratch/niangxiu/fds/apps/change_u_finer_reso')
BASE_PATH = os.path.join(MY_PATH, 'charles')
vperp_PATH = os.path.join(MY_PATH, 'vperp')
if os.path.exists(vperp_PATH):
    shutil.rmtree(vperp_PATH)
os.mkdir(vperp_PATH)

REF_PATH      = os.path.join(MY_PATH, 'ref')
REF_DATA_FILE = os.path.join(REF_PATH, 'initial.les')
REF_SUPP_FILE = os.path.join(REF_PATH, 'charles.in')
CHARLES_BIN   = os.path.join(REF_PATH, 'charles.exe')  

checkpoint = load_last_checkpoint(BASE_PATH, total_MODES)
assert verify_checkpoint(checkpoint)
_, _, _, lss, G_lss, g_lss, J, G_dil, g_dil = checkpoint
alpha = lss.solve()

# draw shadowing field
for i, i_segment in enumerate(K_SEGMENTS):
    print('i_segment = ', i_segment)
    checkpoint = load_checkpoint(os.path.join(BASE_PATH, 'm'+str(total_MODES)+'_segment'+str(i_segment)))
    assert verify_checkpoint(checkpoint)
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())
    run_id = 'time_dilation_{0:02d}'.format(i_segment)
    time_dil = TimeDilation(solve, u0, INLET_U_BASE, run_id, simultaneous_runs = 1, interprocess=interprocess)

    V = time_dil.project(V)
    v = time_dil.project(v)    
    vperp = v 
    for j_mode in range(total_MODES):
        vperp += alpha[i_segment,j_mode] * V[j_mode]

    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())
    run_compute([vperp], spawn_compute_job=None, interprocess=interprocess)
    vperp = vperp.field

    # compute norms
    norm_arr[i] = linalg.norm(vperp) / sqrt(m_cells * 3)
    print(norm_arr[i])

    # run_id = 'seg'+str(i_segment)
    # work_path = os.path.join(vperp_PATH, run_id)
    # os.mkdir(work_path)
    # solut_path = os.path.join(work_path, 'SOLUT_2')
    # os.mkdir(solut_path)
    # initial_data_file = os.path.join(work_path, 'initial.les')
    # shutil.copy(REF_DATA_FILE, initial_data_file)
    # shutil.copy(REF_SUPP_FILE, work_path)

    # # save les file for vperp
    # save_compressible_les(initial_data_file, make_data(vperp), verbose=False)

    # # run charles for 0 step
    # outfile = os.path.join(work_path, 'out')
    # with open(outfile, 'w', 8) as f:
        # subprocess.call(['mpiexec', '-n', str(MPI_NP), CHARLES_BIN],
                # cwd=work_path, stdout=f, stderr=f)
    
    # # copy the results file 
    # result_file = os.path.join(solut_path, 'z0_plane.000000.vtu')
    # shutil.copy(result_file, os.path.join(vperp_PATH, 'z0_plane_seg.'+str(i_segment)+'.vtu'))
    # shutil.rmtree(work_path)

# # draw primal field
# for i, i_segment in enumerate(K_SEGMENTS):
    # print('i_segment = ', i_segment)
    # checkpoint = load_checkpoint(os.path.join(BASE_PATH, 'm'+str(total_MODES)+'_segment'+str(i_segment)))
    # assert verify_checkpoint(checkpoint)
    # u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

    # run_id = 'seg'+str(i_segment)
    # work_path = os.path.join(vperp_PATH, run_id)
    # os.mkdir(work_path)
    # solut_path = os.path.join(work_path, 'SOLUT_2')
    # os.mkdir(solut_path)
    # initial_data_file = os.path.join(work_path, 'initial.les')
    # shutil.copy(REF_DATA_FILE, initial_data_file)
    # shutil.copy(REF_SUPP_FILE, work_path)

    # # save les file for vperp
    # save_compressible_les(initial_data_file, make_data(u0.field), verbose=False)

    # # run charles for 0 step
    # outfile = os.path.join(work_path, 'out')
    # with open(outfile, 'w', 8) as f:
        # subprocess.call(['mpiexec', '-n', str(MPI_NP), CHARLES_BIN],
                # cwd=work_path, stdout=f, stderr=f)
    
    # # copy the results file 
    # result_file = os.path.join(solut_path, 'z0_plane.000000.vtu')
    # shutil.copy(result_file, os.path.join(vperp_PATH, 'z0_plane_seg.'+str(i_segment)+'.vtu'))
    # shutil.rmtree(work_path)


print('the array of norm is:')
print(norm_arr)
pickle.dump((K_SEGMENTS, norm_arr), open("norms.p", "wb"))
