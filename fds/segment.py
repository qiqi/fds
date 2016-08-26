from __future__ import division
import os
import numpy as np
import pascal_lite as pascal
from multiprocessing import Pool

def mpi_compute(outputs):
    zero = pascal.builtin.ZERO
    random = pascal.builtin.RANDOM[0]

    graph = pascal.ComputationalGraph([x.value for x in outputs])
    array_inputs = [x for x in graph.input_values if x not in [zero, random]]
    serial_mode = isinstance(array_inputs[0].field, np.ndarray)
    if serial_mode:
        n = array_inputs[0].field.shape[0]
    else:
        n = np.loadtxt(array_inputs[0].field).shape[0]

    def inputs(x):
        if x is zero:
            return np.zeros(n)
        elif x is random:
            shape = random.shape + (n,)
            return np.random.rand(*shape)
        elif serial_mode:
            return x.field
        else:
            return np.loadtxt(x.field)
    actual_outputs = graph(inputs)
    for index, output in enumerate(outputs):
        if serial_mode:
            output.value.field = actual_outputs[index]
        else:
            parent_dir = os.path.dirname(output.field)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            np.savetxt(output.field, actual_outputs[index])
    return outputs

def trapez_mean(J, dim):
    J = np.rollaxis(J, dim)
    J_m1 = 2 * J[0] - J[1]
    return (J.sum(0) + J[:-1].sum(0) + J_m1) / (2 * J.shape[0])

def run_segment(run, u0, V, v, parameter, i_segment, steps,
                epsilon, simultaneous_runs, interprocess,
                get_host_dir):
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
    res_0 = threads.apply_async(
            run, (u0.field, parameter, steps, run_id, interprocess))
    # run homogeneous tangents
    res_h = []
    subspace_dimension = len(V)

    u1i = u0 + v * epsilon
    run_ids = ['segment{0:02d}_param_perturb{1:03d}'.format(
             i_segment, subspace_dimension)]
    u1i.value.field = os.path.join(get_host_dir(run_ids[0]), 'input.fds')
    u1h = []
    for j in range(subspace_dimension):
        u1h.append(u0 + V[j] * epsilon)
        run_ids.append('segment{0:02d}_init_perturb{1:03d}'.format(i_segment, j))
        u1h[-1].value.field = os.path.join(get_host_dir(run_ids[-1]), 'input.fds')

    # compute everything
    u = mpi_compute([u1i] + u1h)
    u1i = u[0]
    u1h = u[1:]

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
