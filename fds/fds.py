import os
import argparse
from copy import deepcopy
from multiprocessing import Manager

from numpy import *

from .checkpoint import Checkpoint, verify_checkpoint, save_checkpoint
from .timedilation import TimeDilation, TimeDilationExact
from .segment import run_segment, trapez_mean
from .lsstan import LssTangent

# ---------------------------------------------------------------------------- #

def tangent_initial_condition(degrees_of_freedom, subspace_dimension):
    random.seed(12)
    W = random.rand(degrees_of_freedom, subspace_dimension)
    W, _ = linalg.qr(W)
    w = zeros(degrees_of_freedom)
    return W, w

def windowed_mean(a):
    win = sin(linspace(0, pi, a.shape[0]+2)[1:-1])**2
    return (a * win[:,newaxis]).sum(0) / win.sum()

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
    grad_lss = (alpha[:,:,newaxis] * array(G_lss)).sum(1) + array(g_lss)
    J = array(J)
    dJ = trapez_mean(J.mean(0), 0) - J[:,-1]
    steps_per_segment = J.shape[1]
    dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
    grad_dil = dil[:,newaxis] * dJ
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)

class RunWrapper:
    def __init__(self, run):
        self.run = run

    def variable_args(self, u0, parameter, steps, run_id, interprocess):
        try:
            return self.run(u0, parameter, steps,
                            run_id=run_id, interprocess=interprocess)
        except TypeError:
            pass # does not expect run_id or interprocess argument
        try:
            return self.run(u0, parameter, steps, run_id=run_id)
        except TypeError:
            pass # does not expect run_id
        try:
            return self.run(u0, parameter, steps, interprocess=interprocess)
        except TypeError:
            pass # expects neither run_id nor interprocess
        return self.run(u0, parameter, steps)

    def __call__(self, u0, parameter, steps, run_id, interprocess):
        u1, J = self.variable_args(u0, parameter, steps, run_id, interprocess)
        return (array(u1).reshape(array(u0).shape),
                array(J).reshape([steps, -1]))

def continue_shadowing(
        run, parameter, checkpoint,
        num_segments, steps_per_segment, epsilon=1E-6,
        checkpoint_path=None, checkpoint_interval=1, simultaneous_runs=None,
        run_ddt=None, return_checkpoint=False):
    """
    """
    run = RunWrapper(run)
    assert verify_checkpoint(checkpoint)
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    run_id = 'time_dilation_{0:02d}'.format(lss.K_segments())
    if run_ddt is not None:
        time_dil = TimeDilationExact(run_ddt, u0, parameter)
    else:
        time_dil = TimeDilation(run, u0, parameter, run_id,
                                simultaneous_runs, interprocess)

    for i in range(lss.K_segments(), num_segments):
        V = time_dil.project(V)
        v = time_dil.project(v)

        u0, V, v, J0, G, g = run_segment(
                run, u0, V, v, parameter, i, steps_per_segment,
                epsilon, simultaneous_runs, interprocess)
        J_hist.append(J0)
        G_lss.append(G)
        g_lss.append(g)

        # time dilation contribution
        run_id = 'time_dilation_{0:02d}'.format(i+1)
        if run_ddt is not None:
            time_dil = TimeDilationExact(run_ddt, u0, parameter)
        else:
            time_dil = TimeDilation(run, u0, parameter, run_id,
                                    simultaneous_runs, interprocess)
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))

        V = time_dil.project(V)
        v = time_dil.project(v)

        lss.checkpoint(V, v)

        checkpoint = Checkpoint(
                u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil)
        print(lss_gradient(checkpoint))
        if checkpoint_path and (i+1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, checkpoint)
    if return_checkpoint:
        return checkpoint
    else:
        G = lss_gradient(checkpoint)
        return array(J_hist).mean((0,1)), G

def shadowing(
        run, u0, parameter, subspace_dimension, num_segments,
        steps_per_segment, runup_steps, epsilon=1E-6,
        checkpoint_path=None, checkpoint_interval=1, simultaneous_runs=None,
        run_ddt=None, return_checkpoint=False):
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
    run = RunWrapper(run)
    manager = Manager()
    interprocess = (manager.Lock(), manager.dict())

    if runup_steps > 0:
        u0, _ = run(u0, parameter, runup_steps, 'runup', interprocess)

    V, v = tangent_initial_condition(u0.size, subspace_dimension)
    lss = LssTangent()
    checkpoint = Checkpoint(u0, V, v, lss, [], [], [], [], [])
    return continue_shadowing(
            run, parameter, checkpoint,
            num_segments, steps_per_segment, epsilon,
            checkpoint_path, checkpoint_interval,
            simultaneous_runs, run_ddt, return_checkpoint)
