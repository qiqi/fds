from __future__ import division
import os
import numpy as np
import pascal_lite as pascal
from multiprocessing import Pool

from .compute import run_compute

def trapez_mean(J, dim):
    J = np.rollaxis(J, dim)
    J_m1 = 2 * J[0] - J[1]
    return (J.sum(0) + J[:-1].sum(0) + J_m1) / (2 * J.shape[0])

def run_segment(run, u0, V, v, parameter, i_segment, steps,
                epsilon, simultaneous_runs, interprocess,
                get_host_dir=None, 
                compute_outputs = None,
                spawn_compute_job = None):
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
    if get_host_dir is None:
        get_host_dir = lambda x: x
    if compute_outputs is None:
        compute_outputs = []

    threads = Pool(simultaneous_runs)
    run_id = 'segment{0:02d}_baseline'.format(i_segment)
    # run homogeneous tangents
    res_h = []
    subspace_dimension = len(V)

    u1i = u0 + v * epsilon
    run_ids = ['segment{0:02d}_param_perturb{1:03d}'.format(
             i_segment, subspace_dimension)]
    u1i.value.field = os.path.join(get_host_dir(run_ids[0]), 'input.h5')
    u1h = []
    for j in range(subspace_dimension):
        u1h.append(u0 + V[j] * epsilon)
        run_ids.append('segment{0:02d}_init_perturb{1:03d}'.format(i_segment, j))
        u1h[-1].value.field = os.path.join(get_host_dir(run_ids[-1]), 'input.h5')

    # compute all outputs
    run_compute([u1i] + u1h + compute_outputs, spawn_compute_job=spawn_compute_job, interprocess=interprocess)

    res_0 = threads.apply_async(
            run, (u0.field, parameter, steps, run_id, interprocess))

    for j in range(subspace_dimension):
        res_h.append(threads.apply_async(
            run, (u1h[j].field, parameter, steps, run_ids[1+j], interprocess)))
    # run inhomogeneous tangent
    res_i = threads.apply_async(
            run, (u1i.field, parameter + epsilon, steps, run_ids[0], interprocess))

    u0p, J0 = res_0.get()
    u0p = pascal.symbolic_array(field=u0p)
    # get homogeneous tangents
    G = []
    V = pascal.random(subspace_dimension)
    for j in range(subspace_dimension):
        u1p, J1 = res_h[j].get()
        u1p = pascal.symbolic_array(field=u1p)
        V[j] = (u1p - u0p) / epsilon
        G.append(trapez_mean(J1 - J0, 0) / epsilon)
    # get inhomogeneous tangent
    u1p, J1 = res_i.get()
    u1p = pascal.symbolic_array(field=u1p)
    v, g = (u1p - u0p) / epsilon, trapez_mean(J1 - J0, 0) / epsilon
    threads.close()
    threads.join()
    return u0p, V, v, J0, G, g
