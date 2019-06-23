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
from charles import *

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import *
rcParams.update({'axes.labelsize':'xx-large'})
rcParams.update({'xtick.labelsize':'xx-large'})
rcParams.update({'ytick.labelsize':'xx-large'})
rcParams.update({'legend.fontsize':'xx-large'})
rc('font', family='sans-serif')

sys.path.append("/scratch/niangxiu/fds_4CLV_finer_reso")
from fds import *
from fds.checkpoint import *
from fds.cti_restart_io import *
from fds.compute import run_compute
sys.setrecursionlimit(12000)

M_MODES = array([0,7,16,39])
total_MODES = 40
K_SEGMENTS = (400,)
# K_SEGMENTS = range(350, 451)
MPI_NP = 1

MY_PATH = os.path.abspath('/scratch/niangxiu/fds_4CLV_finer_reso/apps')
BASE_PATH = os.path.join(MY_PATH, 'charles')
CLV_PATH = os.path.join(MY_PATH, 'CLV')
if os.path.exists(CLV_PATH):
    shutil.rmtree(CLV_PATH)
os.mkdir(CLV_PATH)
RESULT_PATH = []
for j in M_MODES:
    RESULT_PATH.append(os.path.join(CLV_PATH, 'CLV'+str(j)))
    os.mkdir(RESULT_PATH[-1])

REF_PATH      = os.path.join(MY_PATH, 'ref')
REF_DATA_FILE = os.path.join(REF_PATH, 'initial.les')
REF_SUPP_FILE = os.path.join(REF_PATH, 'charles.in')
CHARLES_BIN   = os.path.join(REF_PATH, 'charles.exe')  

checkpoint = load_last_checkpoint(BASE_PATH, total_MODES)
assert verify_checkpoint(checkpoint)
C = checkpoint.lss.lyapunov_covariant_vectors()
# someone evilly rolled the axis in the lyapunov_covariant_vectors function
C = rollaxis(C,2)
C = rollaxis(C,2) # now the shape is [K_SEGMENTS, M_MODES, M_MODES]
I = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 9, 12, 13, 14, 15, 18, 17, 16, 22, 21, 19, 20, 23, 24, 26, 25, 31, 27, 28, 30, 35, 29, 37, 33, 38, 36, 34, 32, 39]
I = array(I)
print('C.shape = ', C.shape)


def les2vtu(work_path, solut_path, j):
    # run charles for 0 step
    outfile = os.path.join(work_path, 'out')
    with open(outfile, 'w', 8) as f:
        subprocess.call(['mpiexec', '-n', str(MPI_NP), charles_bin],
                cwd=work_path, stdout=f, stderr=f)

    # copy the results file and delete the working folder
    result_file = os.path.join(solut_path, 'z0_plane.000000.vtu')
    shutil.copy(result_file, os.path.join(RESULT_PATH[j], 'z0_plane_seg.'+str(i_segment)+'.vtu'))
    shutil.rmtree(work_path)
    

# draw CLV field
for i_segment in K_SEGMENTS:
    print('i_segment = ', i_segment)
    checkpoint = load_checkpoint(os.path.join(BASE_PATH, 'm'+str(total_MODES)+'_segment'+str(i_segment)))
    assert verify_checkpoint(checkpoint)
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint
    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())
    run_compute([V], spawn_compute_job=None, interprocess=interprocess)
    V = V.field
    print(V.shape)

    # construct CLV at this segment
    CLV = dot(V.T, C[i_segment, :, :])
    CLV = CLV.T

    # plot CLV
    for j, j_mode in enumerate(I[M_MODES]):
        run_id = 'CLV'+str(j_mode)+'_seg'+str(i_segment)
        print('runid:', run_id)
        work_path = os.path.join(CLV_PATH, run_id)
        os.mkdir(work_path)
        solut_path = os.path.join(work_path, 'SOLUT_2')
        os.mkdir(solut_path)
        initial_data_file = os.path.join(work_path, 'initial.les')
        shutil.copy(REF_DATA_FILE, initial_data_file)
        shutil.copy(REF_SUPP_FILE, work_path)

        print(CLV[j_mode].shape)
        save_compressible_les_normalized(initial_data_file, make_data(CLV[j_mode]), verbose=False)
        les2vtu(work_path, solut_path, j)

    # plot flow field
    run_id = 'primal'+'_seg'+str(i_segment)
    print('runid:', run_id)
    work_path = os.path.join(CLV_PATH, run_id)
    os.mkdir(work_path)
    solut_path = os.path.join(work_path, 'SOLUT_2')
    os.mkdir(solut_path)
    initial_data_file = os.path.join(work_path, 'initial.les')
    shutil.copy(REF_DATA_FILE, initial_data_file)
    shutil.copy(REF_SUPP_FILE, work_path)
    
    # print('u0 shape: ', u0.field.shape)
    save_compressible_les(initial_data_file, make_data(u0.field), verbose=False)
    les2vtu(work_path, solut_path, j)
