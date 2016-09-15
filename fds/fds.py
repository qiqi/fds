import os
import sys
import argparse
from copy import deepcopy
import traceback
from multiprocessing import Manager

import pascal_lite as pascal
import numpy as np

from .checkpoint import Checkpoint, verify_checkpoint, save_checkpoint
from .timedilation import TimeDilation, TimeDilationExact
from .segment import run_segment, trapez_mean
from .lsstan import LssTangent#, tangent_initial_condition
from .timeseries import windowed_mean
from .compute import run_compute

# ---------------------------------------------------------------------------- #

def tangent_initial_condition(subspace_dimension):
    #np.random.seed(12)
    W = pascal.random(subspace_dimension)
    #W = (pascal.qr(W.T))[0].T
    W = pascal.qr_transpose(W)[0]
    w = pascal.zeros()
    return W, w

def lss_gradient(checkpoint, num_segments=None):
    _, _, _, lss, G_lss, g_lss, J, G_dil, g_dil = checkpoint
    if num_segments:
        lss = deepcopy(lss)
        lss.bs = lss.bs[:num_segments]
        lss.Rs = lss.Rs[:num_segments]
        G_lss = G_lss[:num_segments]
        g_lss = g_lss[:num_segments]
        J = J[:num_segments]
        G_dil = G_dil[:num_segments]
        g_dil = g_dil[:num_segments]
    alpha = lss.solve()
    grad_lss = (alpha[:,:,np.newaxis] * np.array(G_lss)).sum(1) + np.array(g_lss)
    J = np.array(J)
    dJ = trapez_mean(J.mean(0), 0) - J[:,-1]
    steps_per_segment = J.shape[1]
    dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
    grad_dil = dil[:,np.newaxis] * dJ
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)

class RunWrapper:
    def __init__(self, run):
        self.run = run

    def variable_args(self, u0, parameter, steps, run_id, interprocess):
        try:
            return self.run(u0, parameter, steps,
                            run_id=run_id, interprocess=interprocess)
        except TypeError as e1:
            # does not expect run_id or interprocess argument
            tb1 = traceback.format_exc()
        try:
            return self.run(u0, parameter, steps, run_id=run_id)
        except TypeError as e2:
            # does not expect run_id
            tb2 = traceback.format_exc()
        try:
            return self.run(u0, parameter, steps, interprocess=interprocess)
        except TypeError as e3:
            # expects neither run_id nor interprocess
            tb3 = traceback.format_exc()
        try:
            return self.run(u0, parameter, steps)
        except TypeError as e4:
            tb4 = traceback.format_exc() # failed
        for tb in (tb1, tb2, tb3, tb4):
            sys.stderr.write(str(tb) + '\n')
        raise TypeError

    def __call__(self, u0, parameter, steps, run_id, interprocess):
        try:
            u1, J = self.variable_args(
                    u0, parameter, steps, run_id, interprocess)
            return u1, np.array(J).reshape([steps, -1])
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(str(tb) + '\n')
            raise e

def continue_shadowing(
        run, parameter, checkpoint,
        num_segments, steps_per_segment, epsilon=1E-6,
        checkpoint_path=None, checkpoint_interval=1, simultaneous_runs=None,
        run_ddt=None, return_checkpoint=False, get_host_dir=None, spawn_compute_job=None):
    """
    """
    compute_outputs = []

    run = RunWrapper(run)
    assert verify_checkpoint(checkpoint)
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    i = lss.K_segments()
    run_id = 'time_dilation_{0:02d}'.format(i)
    if run_ddt is not None:
        time_dil = TimeDilationExact(run_ddt, u0, parameter)
    else:
        time_dil = TimeDilation(run, u0, parameter, run_id,
                                simultaneous_runs, interprocess)

    V = time_dil.project(V)
    v = time_dil.project(v)

    u0, V, v, J0, G, g = run_segment(
            run, u0, V, v, parameter, i, steps_per_segment,
            epsilon, simultaneous_runs, interprocess, get_host_dir=get_host_dir,
            compute_outputs=compute_outputs, spawn_compute_job=spawn_compute_job)

    J_hist.append(J0)
    G_lss.append(G)
    g_lss.append(g)

    for i in range(lss.K_segments() + 1, num_segments + 1):

        # time dilation contribution
        run_id = 'time_dilation_{0:02d}'.format(i)
        if run_ddt is not None:
            time_dil = TimeDilationExact(run_ddt, u0, parameter)
        else:
            time_dil = TimeDilation(run, u0, parameter, run_id,
                                    simultaneous_runs, interprocess)
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))

        V = time_dil.project(V)
        v = time_dil.project(v)

        V, v = lss.checkpoint(V, v)
        # extra outputs to compute
        compute_outputs = [lss.Rs[-1], lss.bs[-1], G_dil[-1], g_dil[-1]]

        # run all segments
        if i < num_segments:
            u0, V, v, J0, G, g = run_segment(
                    run, u0, V, v, parameter, i, steps_per_segment,
                    epsilon, simultaneous_runs, interprocess, get_host_dir=get_host_dir,
                    compute_outputs=compute_outputs, spawn_compute_job=spawn_compute_job)
        else:
            run_compute(compute_outputs, spawn_compute_job=spawn_compute_job, interprocess=interprocess)

        for output in [lss.Rs, lss.bs, G_dil, g_dil]:
            output[-1] = output[-1].field

        checkpoint = Checkpoint(
                u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil)
        print(lss_gradient(checkpoint))
        sys.stdout.flush()

        if checkpoint_path and (i) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, checkpoint)

        if i < num_segments:
            J_hist.append(J0)
            G_lss.append(G)
            g_lss.append(g)

    if return_checkpoint:
        return checkpoint
    else:
        G = lss_gradient(checkpoint)
        return np.array(J_hist).mean((0,1)), G

def shadowing(
        run, u0, parameter, subspace_dimension, num_segments,
        steps_per_segment, runup_steps, epsilon=1E-6,
        checkpoint_path=None, checkpoint_interval=1, simultaneous_runs=None,
        run_ddt=None, return_checkpoint=False, get_host_dir=None, spawn_compute_job=None):
    '''
    run: a function in the form
         u1, J = run(u0, parameter, steps, run_id, interprocess)

         inputs  - u0:           init solution, a flat numpy array of doubles.
                   parameter:    design parameter, a single number.
                   steps:        number of time steps, an int.
                   run_id:       a unique identifier, a string,
                                 e.g., "segment02_init_perturb003".
                   interprocess: a tuple of (lock, dict) for
                                 synchronizing between different runs.
                                 lock: a multiprocessing.Manager.Lock object.
                                 dict: a multiprocessing.Manager.dict object.
         outputs - u1:           final solution, a flat numpy array of doubles,
                                 must be of the same size as u0.
                   J:            quantities of interest, a numpy array of shape
                                 (steps, n_qoi), where n_qoi is an arbitrary
                                 but consistent number, # quantities of interest.
    '''
    u0 = pascal.symbolic_array(field=u0)

    run = RunWrapper(run)
    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    if runup_steps > 0:
        u0, _ = run(u0.field, parameter, runup_steps, 'runup', interprocess)
        u0 = pascal.symbolic_array(field=u0)
        #print u0.field

    V, v = tangent_initial_condition(subspace_dimension)
    lss = LssTangent()
    checkpoint = Checkpoint(u0, V, v, lss, [], [], [], [], [])
    return continue_shadowing(
            run, parameter, checkpoint,
            num_segments, steps_per_segment, epsilon,
            checkpoint_path, checkpoint_interval,
            simultaneous_runs, run_ddt, return_checkpoint, get_host_dir, spawn_compute_job)
