import os
import argparse
from subprocess import *
from multiprocessing import Pool, Manager

from numpy import *
from scipy import sparse
import scipy.sparse.linalg as splinalg

# ---------------------------------------------------------------------------- #

def tangent_initial_condition(degrees_of_freedom, subspace_dimension):
    random.seed(12)
    W = random.rand(degrees_of_freedom, subspace_dimension)
    W, _ = linalg.qr(W)
    w = zeros(degrees_of_freedom)
    return W, w

class TimeDilation:
    def __init__(self, run, u0, parameter, run_id, lock):
        dof = u0.size
        u0p, _ = run(u0, parameter, 1, run_id, lock)
        self.dxdt = (u0p - u0)   # time step size turns out cancelling out
        self.dxdt_normalized = self.dxdt / linalg.norm(self.dxdt)

    def contribution(self, v):
        return dot(self.dxdt, v) / (self.dxdt**2).sum()

    def project(self, v):
        dv = outer(self.dxdt_normalized, dot(self.dxdt_normalized, v))
        return v - dv.reshape(v.shape)

class LssTangent:
    def __init__(self):
        self.Rs = []
        self.bs = []

    def checkpoint(self, V, v):
        Q, R = linalg.qr(V)
        b = dot(Q.T, v)
        self.Rs.append(R)
        self.bs.append(b)
        V[:] = Q
        v -= dot(Q, b)

    def solve(self):
        Rs, bs = array(self.Rs), array(self.bs)
        assert Rs.ndim == 3 and bs.ndim == 2
        assert Rs.shape[0] == bs.shape[0]
        assert Rs.shape[1] == Rs.shape[2] == bs.shape[1]
        nseg, subdim = bs.shape
        eyes = eye(subdim, subdim) * ones([nseg, 1, 1])
        matrix_shape = (subdim * nseg, subdim * (nseg+1))
        I = sparse.bsr_matrix((eyes, r_[1:nseg+1], r_[:nseg+1]))
        D = sparse.bsr_matrix((Rs, r_[:nseg], r_[:nseg+1]), shape=matrix_shape)
        B = (D - I).tocsr()
        alpha = -(B.T * splinalg.spsolve(B * B.T, ravel(bs)))
        return alpha.reshape([nseg+1,-1])[:-1]

# -------------------------------- main loop --------------------------------- #

def run_segment(run, u0, V, v, parameter, i_segment, steps,
                epsilon, simultaneous_runs, lock):
    '''
    Run Time Segement i_segment, starting from
        u0: nonlinear solution
        V:  homogeneous tangents
        v:  inhomogeneous tangent
    for steps time steps, and returns afterwards
        u0: nonlinear solution
        V:  homogeneous tangents
        v:  inhomogeneous tangent
        J0: history of quantities of interest for the nonlinear solution
        G:  sensitivities of the homogeneous tangents
        g:  sensitivity of the inhomogeneous tangent
    '''
    threads = Pool(simultaneous_runs)
    run_id = 'segment{0:02d}_baseline'.format(i_segment)
    res_0 = threads.apply_async(run, (u0, parameter, steps, run_id, lock))
    # run homogeneous tangents
    res_h = []
    subspace_dimension = V.shape[1]
    for j in range(subspace_dimension):
        u1 = u0 + V[:,j] * epsilon
        run_id = 'segment{0:02d}_init_perturb{1:03d}'.format(i_segment, j)
        res_h.append(threads.apply_async(
            run, (u1, parameter, steps, run_id, lock)))
    # run inhomogeneous tangent
    u1 = u0 + v * epsilon
    run_id = 'segment{0:02d}_param_perturb'.format(i_segment)
    res_i = threads.apply_async(
            run, (u1, parameter + epsilon, steps, run_id, lock))

    u0p, J0 = res_0.get()
    # get homogeneous tangents
    G = []
    for j in range(subspace_dimension):
        u1p, J1 = res_h[j].get()
        V[:,j] = (u1p - u0p) / epsilon
        G.append((J1.mean(0) - J0.mean(0)) / epsilon)
    # get inhomogeneous tangent
    u1p, J1 = res_i.get()
    v, g = (u1p - u0p) / epsilon, (J1.mean(0) - J0.mean(0)) / epsilon
    threads.close()
    threads.join()
    return u0p, V, v, J0, G, g

def windowed_mean(a):
    win = sin(linspace(0, pi, a.shape[0]+2)[1:-1])**2
    return (a * win[:,newaxis]).sum(0) / win.sum()

def lss_gradient(lss, G_lss, g_lss, J, G_dil, g_dil):
    alpha = lss.solve()
    grad_lss = (alpha[:,:,newaxis] * array(G_lss)).sum(1) + array(g_lss)
    J = array(J)
    dJ = J.mean((0,1)) - J[:,-1]
    steps_per_segment = J.shape[1]
    dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
    grad_dil = dil[:,newaxis] * dJ
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)

def print_info(verbose, grad_hist, lss, G_lss, g_lss, J_hist, G_dil, g_dil):
    if verbose:
        print('LSS gradient = ', grad_hist[-1])
    if isinstance(verbose, str):
        savez('lss.npz',
              G_lss=G_lss,
              g_lss=g_lss,
              G_dil=G_dil,
              g_dil=g_dil,
              R=lss.Rs,
              b=lss.bs,
              grad_hist=grad_hist
        )

def finite_difference_shadowing(
        run, u0, parameter, subspace_dimension, num_segments,
        steps_per_segment, runup_steps, epsilon=1E-6, verbose=0,
        simultaneous_runs=None):
    '''
    run: a function in the form
         u1, J = run(u0, parameter, steps, run_id, lock)

         inputs  - u0:        initial solution, a flat numpy array of doubles.
                   parameter: design parameter, a single number.
                   steps:     number of time steps, an int.
                   run_id:    a unique identifier, a string,
                              e.g., "segment02_init_perturb003".
                   lock:      a tuple of (lock, dict) for synchronizing
                              between different runs,
                              a multiprocessing.Lock object.
         outputs - u1:        final solution, a flat numpy array of doubles,
                              must be of the same size as u0.
                   J:         quantities of interest, a numpy array of shape
                              (steps, n_qoi), where n_qoi is an arbitrary
                              but consistent number, # quantities of interest.
    '''

    manager = Manager()
    lock = (manager.Lock(), manager.dict())

    degrees_of_freedom = u0.size

    J_hist = []
    G_lss = []
    g_lss = []
    G_dil = []
    g_dil = []
    grad_hist = []

    if runup_steps > 0:
        u0, _ = run(u0, parameter, runup_steps, 'runup', lock)
    time_dil = TimeDilation(
            run, u0, parameter, 'time_dilation_initial', lock)

    V, v = tangent_initial_condition(degrees_of_freedom, subspace_dimension)
    lss = LssTangent()
    for i in range(num_segments):
        V = time_dil.project(V)
        v = time_dil.project(v)

        u0, V, v, J0, G, g = run_segment(
                run, u0, V, v, parameter, i, steps_per_segment,
                epsilon, simultaneous_runs, lock)
        J_hist.append(J0)
        G_lss.append(G)
        g_lss.append(g)

        # time dilation contribution
        time_dil = TimeDilation(run, u0, parameter,
                                'time_dilation_{0:02d}'.format(i), lock)
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))
        lss.checkpoint(V, v)

        grad_hist.append(lss_gradient(lss, G_lss, g_lss, J_hist, G_dil, g_dil))
        print_info(verbose, grad_hist, lss, G_lss, g_lss, J_hist, G_dil, g_dil)

    return array(J_hist).mean((0,1)), grad_hist[-1]
