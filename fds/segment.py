from numpy import *
from multiprocessing import Pool

def trapez_mean(J, dim):
    J = rollaxis(J, dim)
    J_m1 = 2 * J[0] - J[1]
    return (J.sum(0) + J[:-1].sum(0) + J_m1) / (2 * J.shape[0])

def run_segment(run, u0, V, v, parameter, i_segment, steps,
                epsilon, simultaneous_runs, interprocess):
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
            run, (u0, parameter, steps, run_id, interprocess))
    # run homogeneous tangents
    res_h = []
    subspace_dimension = V.shape[1]
    for j in range(subspace_dimension):
        u1 = u0 + V[:,j] * epsilon
        run_id = 'segment{0:02d}_init_perturb{1:03d}'.format(i_segment, j)
        res_h.append(threads.apply_async(
            run, (u1, parameter, steps, run_id, interprocess)))
    # run inhomogeneous tangent
    u1 = u0 + v * epsilon
    run_id = 'segment{0:02d}_param_perturb{1:03d}'.format(
             i_segment, subspace_dimension)
    res_i = threads.apply_async(
            run, (u1, parameter + epsilon, steps, run_id, interprocess))

    u0p, J0 = res_0.get()
    # get homogeneous tangents
    G = []
    for j in range(subspace_dimension):
        u1p, J1 = res_h[j].get()
        V[:,j] = (u1p - u0p) / epsilon
        G.append(trapez_mean(J1 - J0, 0) / epsilon)
    # get inhomogeneous tangent
    u1p, J1 = res_i.get()
    v, g = (u1p - u0p) / epsilon, trapez_mean(J1 - J0, 0) / epsilon
    threads.close()
    threads.join()
    return u0p, V, v, J0, G, g
