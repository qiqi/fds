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


# compute CLV, result in degree
def angle(u,v):
    return arccos(dot(u,v) / linalg.norm(u)/ linalg.norm(v)) /pi * 180

# compute angles
K_SEGMENTS = arange(350, 450)
angles = zeros([size(K_SEGMENTS), total_MODES, total_MODES])
for jj1, j1 in enumerate(I): # sorted according to LE
    for jj2, j2 in enumerate(I):
        for i, i_segment in enumerate(K_SEGMENTS):
            angles[i,jj1,jj2] = angle(C[i_segment,:,j1], C[i_segment,:,j2])
            angles[i,jj1,jj2] =  90 - abs(90-angles[i,j1,j2])
            if isnan(angles[i,jj1,jj2]) and jj1!=jj2:
                print('get Nan at ', jj1, jj2)
            if (angles[i,jj1,jj2] <= 10) and jj1!=jj2:
                print('get a small angle of degree ', angles[i,jj1,jj2], 'at i_segment, j1, j2:', i_segment, jj1, jj2)

f = open('angles.txt', 'w')
for j1 in range(total_MODES):
    for j2 in range (total_MODES):
        f.write(j1, j2, mean(angles[:, j1,j2]))

# plot smallest and averaged angles between all pairs of CLVs
smallest_angles = zeros([total_MODES, total_MODES])
averaged_angles = zeros([total_MODES, total_MODES])
for j1 in range(total_MODES):
    for j2 in range(total_MODES):
        smallest_angles[j1,j2] = angles[:,j1,j2].min()
        averaged_angles[j1,j2] = mean(angles[:,j1,j2])
X, Y = meshgrid(range(total_MODES), range(total_MODES))

figure(figsize=(7,4))
contour(X,Y,smallest_angles)
xlabel('# CLV')
ylabel('# CLV')
ax.set_title('Smallest angles over all segments')
tight_layout()
savefig('smallest_angles.png')
close() 


# # plot angle statistics among all CLVs
# angles_new = [angles[i,j1,j2] for i in range(angles.shape[0]) for j1 in range(total_MODES) for j2 in range(total_MODES) if j1 > j2]
# angles_new = array(angles_new)
# print(angles_new.shape)
# print(angles_new.max(), angles_new.min())
# figure(figsize=(7,4))
# hist(angles_new, bins = linspace(0, 180, 61), normed=1, color='grey')
# xlim([0,100])
# xlabel('angles ($\degree$)')
# tight_layout()
# savefig('CLV_finer_angles_all.png')
# close()

# # plot angle statistics among CLVs 5 apart
# angles_new = [angles[i,j1,j2] for i in range(angles.shape[0]) for j1 in range(total_MODES) for j2 in range(total_MODES) if j1 >= j2+5]
# angles_new = array(angles_new)
# print(angles_new.shape)
# print(angles_new.max(), angles_new.min())
# figure(figsize=(7,4))
# hist(angles_new, bins = linspace(0, 180, 61), normed=1, color='grey')
# xlim([0,100])
# xlabel('angles ($\degree$)')
# tight_layout()
# savefig('CLV_finer_angles_5apart.png')
# close()

# # plot angle statistics among neutral and unstable modes
# angles_new = [angles[i,j1,j2] for i in range(angles.shape[0]) for j1 in [16] for j2 in range(total_MODES) if j1 > j2]
# angles_new = array(angles_new)
# print(angles_new.shape)
# print(angles_new.max(), angles_new.min())
# hist(angles_new, bins = linspace(0, 180, 61), normed=1)
# xlim([0,100])
# xlabel('angles ($\degree$)')
# tight_layout()
# savefig('CLV_angles_0and+.png')
# close()

# # plot angle statistics among neutral and stable modes
# angles_new = [angles[i,j1,j2] for i in range(angles.shape[0]) for j1 in [16] for j2 in range(total_MODES) if j1 < j2]
# angles_new = array(angles_new)
# print(angles_new.shape)
# print(angles_new.max(), angles_new.min())
# hist(angles_new, bins = linspace(0, 180, 61), normed=1)
# xlim([0,100])
# xlabel('angles ($\degree$)')
# tight_layout()
# savefig('CLV_angles_0and-.png')
# close()
