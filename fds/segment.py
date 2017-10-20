from __future__ import division
import os
import numpy as np
import pascal_lite as pascal
from multiprocessing import Pool

from .compute import run_compute

def trapez_mean(J, dim):
    J = np.rollaxis(J, dim)
    return (J[1:].sum(0) + J[:-1].sum(0)) / (2 * (J.shape[0] - 1))

def run_segment(run, u0, V, v, parameter, i_segment, steps,
                epsilon, simultaneous_runs):
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
    baseline_run = threads.apply_async(run, (u0, parameter, steps, run_id))

    homogeneous_runs = []
    for j, Vj in enumerate(V):
        run_id = 'segment{0:02d}_init_pert{1:03d}'.format(i_segment, j)
        homogeneous_runs.append(threads.apply_async(
            run, (u0 + Vj * epsilon, parameter, steps, run_id)))

    u1i = u0 + v * epsilon
    run_id = 'segment{0:02d}_param_pert{1:03d}'.format(i_segment, len(V))
    inhomogeneous_run = threads.apply_async(
            run, (u0 + v * epsilon, parameter + epsilon, steps, run_id))

    u0p, J0 = baseline_run.get()
    V, G = [], []
    for j, run in enumerate(homogeneous_runs):
        u1p, J1 = run.get()
        V.append((u1p - u0p) / epsilon)
        G.append(trapez_mean(J1 - J0, 0) / epsilon)
    u1p, J1 = inhomogeneous_run.get()
    v, g = (u1p - u0p) / epsilon, trapez_mean(J1 - J0, 0) / epsilon
    threads.close()
    threads.join()
    return u0p, V, v, J0, G, g

def tangent_segment(tangent_run, u0, V, v, parameter, i_segment, steps,
                    simultaneous_runs):
    threads = Pool(simultaneous_runs)
    homogeneous_runs = []
    for j, Vj in enumerate(V):
        run_id = 'segment{0:02d}_init_tan{1:03d}'.format(i_segment, j)
        homogeneous_runs.append(threads.apply_async(
            tangent_run, (u0, parameter, Vj, 0, steps, run_id)))

    run_id = 'segment{0:02d}_param_tan{1:03d}'.format(i_segment, len(V))
    inhomogeneous_run = threads.apply_async(
            tangent_run, (u0, parameter, v, 1, steps, run_id))

    V, G = [], []
    for j, run in enumerate(homogeneous_runs):
        u0p, vp, J0, dJ = run.get()
        V.append(vp)
        G.append(trapez_mean(dJ, 0))
    u0p, vp, J0, dJ = inhomogeneous_run.get()
    v, g = vp, trapez_mean(dJ, 0)
    threads.close()
    threads.join()
    return u0p, V, v, J0, G, g

def adjoint_segment(adjoint_run, u0, w, parameter, i_segment, steps, weight):
    run_id = 'segment{0:02d}_adjoint'.format(i_segment)
    w /= weight
    wp, dJds = adjoint_run(u0, parameter, steps, w, run_id)
    return wp * weight, dJds * weight
