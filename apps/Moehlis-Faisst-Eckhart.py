# import standard packages
import numpy as np
import scipy as sp
import math as ma
import os
import sys
import time
import numba

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *

u0 = np.array([1.,          # base flow
               0.,
               0.187387,    # streamwise vortex
               0.040112,    # spanwise flow
               0.047047,    # spanwise flow
               0.,
               0.,
               0.013188,    # fully three-dimensional mode
               0.])

@numba.jitclass([('Re', numba.float32),
                 ('Lx', numba.float32),
                 ('Lz', numba.float32),
                 ('c', numba.float64[:])])
class MoehlisFaisstEckhart(object):
    def __init__(self, Re, Lx, Lz):
        a = 2.*np.pi/Lx
        b = np.pi/2.
        c = 2.*np.pi/Lz

        kabc = np.sqrt(a*a + b*b + c*c)
        kac  = np.sqrt(a*a + c*c)
        kbc  = np.sqrt(b*b + c*c)

        # coefficients for first equation
        c1_1 =  b*b/Re
        c1_2 = -np.sqrt(3./2.)*b*c/kabc
        c1_3 =  np.sqrt(3./2.)*b*c/kbc

        # coefficients for second equation
        c2_1 = -(4.*b*b/3. + c*c)/Re
        c2_2 =  5./3.*np.sqrt(2./3.)*c*c/kac
        c2_3 = -c*c/np.sqrt(6.)/kac
        c2_4 = -a*b*c/np.sqrt(6.)/kac/kabc
        c2_5 = -np.sqrt(3./2)*b*c/kbc

        # coefficients for third equation
        c3_1 = -(b*b + c*c)/Re
        c3_2 =  2./np.sqrt(6.)*a*b*c/kac/kbc
        c3_3 =  (b*b*(3.*a*a+c*c) \
                   - 3.*c*c*(a*a+c*c))/np.sqrt(6.)/kac/kbc/kabc

        # coefficients for fourth equation
        c4_1 = -(3*a*a + 4.*b*b)/3./Re
        c4_2 = -a/np.sqrt(6.)
        c4_3 = -10./3./np.sqrt(6.)*a*a/kac
        c4_4 = -np.sqrt(3./2.)*a*b*c/kac/kbc
        c4_5 = -np.sqrt(3./2.)*a*a*b*b/kac/kbc/kabc
        c4_6 = -a/np.sqrt(6.)

        # coefficients for fifth equation
        c5_1 = -(a*a + b*b)/Re
        c5_2 =  a/np.sqrt(6.)
        c5_3 =  a*a/np.sqrt(6.)/kac
        c5_4 = -a*b*c/np.sqrt(6.)/kac/kabc
        c5_5 =  a/np.sqrt(6.)
        c5_6 =  2./np.sqrt(6.)*a*b*c/kac/kbc

        # coefficients for sixth equation
        c6_1 = -(3.*a*a + 4.*b*b + 3.*c*c)/3./Re
        c6_2 =  a/np.sqrt(6.)
        c6_3 =  np.sqrt(3./2.)*b*c/kabc
        c6_4 =  10./3.*(a*a-c*c)/np.sqrt(6.)/kac
        c6_5 = -2.*np.sqrt(2./3.)*a*b*c/kac/kbc
        c6_6 =  a/np.sqrt(6.)
        c6_7 =  np.sqrt(3./2.)*b*c/kabc

        # coefficients for seventh equation
        c7_1 = -(a*a + b*b + c*c)/Re
        c7_2 = -a/np.sqrt(6.)
        c7_3 =  (c*c-a*a)/np.sqrt(6.)/kac
        c7_4 =  a*b*c/np.sqrt(6.)/kac/kbc

        # coefficients for eighth equation
        c8_1 = -(a*a + b*b + c*c)/Re
        c8_2 =  2./np.sqrt(6.)*a*b*c/kac/kabc
        c8_3 =  c*c*(3.*a*a-b*b+3.*c*c)/np.sqrt(6.)/kac/kbc/kabc

        # coefficients for ninth equation
        c9_1 = -9.*b*b/Re
        c9_2 =  np.sqrt(3./2.)*b*c/kbc
        c9_3 = -np.sqrt(3./2.)*b*c/kabc

        self.c = np.array([c1_1, c1_2, c1_3, c2_1, c2_2, c2_3, c2_4, c2_5,
                           c3_1, c3_2, c3_3, c4_1, c4_2, c4_3, c4_4, c4_5,
                           c4_6, c5_1, c5_2, c5_3, c5_4, c5_5, c5_6, c6_1,
                           c6_2, c6_3, c6_4, c6_5, c6_6, c6_7, c7_1, c7_2,
                           c7_3, c7_4, c8_1, c8_2, c8_3, c9_1, c9_2, c9_3])

    def step(self, a, dt):
        (c1_1, c1_2, c1_3, c2_1, c2_2, c2_3, c2_4, c2_5,
         c3_1, c3_2, c3_3, c4_1, c4_2, c4_3, c4_4, c4_5,
         c4_6, c5_1, c5_2, c5_3, c5_4, c5_5, c5_6, c6_1,
         c6_2, c6_3, c6_4, c6_5, c6_6, c6_7, c7_1, c7_2,
         c7_3, c7_4, c8_1, c8_2, c8_3, c9_1, c9_2, c9_3) = self.c

        a1,a2,a3,a4,a5,a6,a7,a8,a9 = a
        L1 = 1. + dt*c1_1
        N1 = c1_2*a6*a8 + \
             c1_3*a2*a3

        L2 = 1. - dt*c2_1
        N2 = c2_2*a4*a6 + \
             c2_3*a5*a7 + \
             c2_4*a5*a8 + \
             c2_5*a1*a3 + \
             c2_5*a3*a9

        L3 = 1. - dt*c3_1
        N3 = c3_2*(a4*a7 + a5*a6) + \
             c3_3*a4*a8

        L4 = 1. - dt*c4_1
        N4 = c4_2*a1*a5 + \
             c4_3*a2*a6 + \
             c4_4*a3*a7 + \
             c4_5*a3*a8 + \
             c4_6*a5*a9

        L5 = 1. - dt*c5_1
        N5 = c5_2*a1*a4 + \
             c5_3*a2*a7 + \
             c5_4*a2*a8 + \
             c5_5*a4*a9 + \
             c5_6*a3*a6

        L6 = 1. - dt*c6_1
        N6 = c6_2*a1*a7 + \
             c6_3*a1*a8 + \
             c6_4*a2*a4 + \
             c6_5*a3*a5 + \
             c6_6*a7*a9 + \
             c6_7*a8*a9

        L7 = 1. - dt*c7_1
        N7 = c7_2*(a1*a6 + a6*a9) + \
             c7_3*a2*a5 + \
             c7_4*a3*a4

        L8 = 1. - dt*c8_1
        N8 = c8_2*a2*a5 + \
             c8_3*a3*a4

        L9 = 1. - dt*c9_1
        N9 = c9_2*a2*a3 + \
             c9_3*a6*a8

        a[0] = (a1 + dt*N1 + dt*c1_1)/L1
        a[1] = (a2 + dt*N2)/L2
        a[2] = (a3 + dt*N3)/L3
        a[3] = (a4 + dt*N4)/L4
        a[4] = (a5 + dt*N5)/L5
        a[5] = (a6 + dt*N6)/L6
        a[6] = (a7 + dt*N7)/L7
        a[7] = (a8 + dt*N8)/L8
        a[8] = (a9 + dt*N9)/L9

    def stepsArray(self, a, dt):
        N = a.shape[0]
        for i in range(1,N):
            a[i,:] = a[i-1,:]
            self.step(a[i,:], dt)

    def steps(self, a0, dt, N):
        a = a0.copy()
        for i in range(1,N):
            self.step(a, dt)
        return a

def MFE(dt,N):
    #==========================================
    # Moehlis-Faisst-Eckhardt model
    #==========================================
    a   = np.zeros((N,9),dtype=float)

    a[0,0] = 1.           # base flow
    a[0,2] = 0.187387     # streamwise vortex
    a[0,3] = 0.040112     # spanwise flow
    a[0,4] = 0.047047     # spanwise flow
    a[0,7] = 0.013188     # fully three-dimensional mode

    mfe = MoehlisFaisstEckhart(Re = 800., Lx = 4.*np.pi, Lz = 2.*np.pi)
    mfe.stepsArray(a, dt)

    return a.T

def solve(u, dt, n):
    return mfe.steps(u, dt, n), zeros(n)

if __name__ == '__main__':
    #==========================================
    # produce and visualize data
    #    Moehlis-Faisst-Eckhart model
    #==========================================
    if not os.path.exists('MFE'):
        os.mkdir('MFE')

    dt  = 1.e-2
    mfe = MoehlisFaisstEckhart(Re = 800., Lx = 4.*np.pi, Lz = 2.*np.pi)
    shadowing(solve, u0,
              0, 9, 20, int(100/dt), int(500/dt), checkpoint_path='MFE')

    '''
    N   = int(np.floor(5000./dt))
    t0  = time.time()
    a1,a2,a3,a4,a5,a6,a7,a8,a9 = MFE(dt,N)
    X   = np.vstack((a1,a2,a3,a4,a5,a6,a7,a8,a9))
    t1  = time.time()
    print('MOEHLIS-FAISST-ECKHARDT system')
    print('generate snapshots (time in sec)    = ',t1-t0)
    print('number of snapshots                 = ',N)

    # visualization
    fig = plt.figure(1)
    ax  = fig.gca(projection='3d')
    istart = 20000
    # streak and roll
    p1  = X[1,istart:]*X[1,istart:] + X[2,istart:]*X[2,istart:]
    # mean flow
    p2  = X[0,istart:]*X[0,istart:] + X[8,istart:]*X[8,istart:]
    # 3D breakdown
    p3  = X[7,istart:]*X[7,istart:]
    ax.plot(p1,p2,p3, 'r',label='self-sustaining process (SSP)')
    ax.legend()
    ax.view_init(45,225)
    ax.set_xlabel('roll & streak')
    ax.set_ylabel('mean shear')
    ax.set_zlabel('burst')
    plt.savefig("MFE.png")
    '''
