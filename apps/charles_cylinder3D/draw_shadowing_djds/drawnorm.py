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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')

K_SEGMENTS, norm_arr = pickle.load(open("norms.p", "rb"))
assert K_SEGMENTS.shape == norm_arr.shape

# normalize
U0 = 33
D = 0.25e-3
Z = 2*D
rho = 1.18
F0 = 0.5 * rho * U0**2 * D * Z
P0 = 0.5 * rho * U0**2
r0 = U0/D
t0 = D/U0
DT = 1e-8 * 200

tt = K_SEGMENTS / (t0/DT)
norm_arr /= 1/U0 

plt.figure(figsize=(7,4))
plt.semilogy(tt, norm_arr)
plt.xlim([64, 94])
plt.ylim([0.3, 50])
plt.xlabel('$T/t_0$')
plt.ylabel('$||v^\perp|| \,/\, U_0^{-1}$')
plt.tight_layout()
plt.savefig('vperp_norm_u')
plt.close()
