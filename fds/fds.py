from __future__ import division
import os
import sys
import argparse
from copy import deepcopy
import traceback
from multiprocessing import Manager

import numpy as np

import pascal_lite as pascal
from .checkpoint import Checkpoint, verify_checkpoint, save_checkpoint
from .timedilation import TimeDilation, TimeDilationExact
from .segment import run_segment, tangent_segment, trapez_mean
from .lsstan import LssTangent#, tangent_initial_condition
from .timeseries import windowed_mean
from .compute import run_compute
from .state import random_states, zero_state, decode_state, encode_state

# ---------------------------------------------------------------------------- #

def tangent_initial_condition(subspace_dimension):
    V = random_states(subspace_dimension)
    v = zero_state()
    return V, v

def lss_gradient(checkpoint):
    _, _, _, lss, G_lss, g_lss, J, G_dil, g_dil = checkpoint
    alpha = lss.solve()
    grad_lss = (alpha[:,:,np.newaxis] * np.array(G_lss)).sum(1) \
             + np.array(g_lss)
    J = np.array(J)
    dJ = trapez_mean(J, 1) - J[:,-1]
    steps_per_segment = J.shape[1] - 1
    dil = ((alpha[:-1] * np.array(G_dil)[1:]).sum(1)
            + np.array(g_dil)[1:]) / steps_per_segment
    grad_dil = dil[:,np.newaxis] * dJ[:-1]
    return windowed_mean(grad_lss) + windowed_mean(grad_dil)

class RunWrapper:
    def __init__(self, run):
        self.run = run

    def variable_args(self, u0, *args, **kwargs):
        u0 = decode_state(u0)
        try:
            return self.run(u0, *args, **kwargs)
        except TypeError as e1:
            # does not expect run_id
            args = args[:-1]
            tb1 = traceback.format_exc()
        try:
            return self.run(u0, *args, **kwargs)
        except TypeError as e2:
            tb2 = traceback.format_exc() # failed
        for tb in (tb1, tb2):
            sys.stderr.write(str(tb) + '\n')
        raise TypeError

    def __call__(self, u, s, steps, *args, **kwargs):
        try:
            u1, J = self.variable_args(u, s, steps, *args, **kwargs)
            return encode_state(u1), np.array(J).reshape([steps+1, -1])
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(str(tb) + '\n')
            raise e

class TangentWrapper(RunWrapper):
    def __init__(self, run):
        self.run = run

    def __call__(self, u, s, du, ds, steps, *args, **kwargs):
        try:
            u, v, J, dJ = self.variable_args(u, s, du, ds, steps,
                                             *args, **kwargs)
            return encode_state(v), encode_state(v), \
                   np.array(J).reshape([steps+1, -1]), \
                   np.array(dJ).reshape([steps+1, -1])
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(str(tb) + '\n')
            raise e

class AdjointWrapper(RunWrapper):
    def __init__(self, run):
        self.run = run

    def __call__(self, u, s, steps, *args, **kwargs):
        try:
            u1, J = self.variable_args(u, s, steps, *args, **kwargs)
            return encode_state(u1), np.array(J)
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(str(tb) + '\n')
            raise e

def continue_shadowing(
        run, parameter, checkpoint,
        num_segments, steps_per_segment, epsilon=1E-6,
        checkpoint_path=None, checkpoint_interval=1, simultaneous_runs=None,
        tangent_run=None, run_ddt=None, return_checkpoint=False):
    """
    """
    run = RunWrapper(run)
    assert verify_checkpoint(checkpoint)
    u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil = checkpoint

    for i in range(lss.K_segments(), num_segments):
        run_id = 'time_dilation_{0:02d}'.format(i)
        if run_ddt is not None:
            time_dil = TimeDilationExact(run_ddt, u0, parameter)
        else:
            time_dil = TimeDilation(run, u0, parameter, run_id,
                                    simultaneous_runs)
        G_dil.append(time_dil.contribution(V))
        g_dil.append(time_dil.contribution(v))
        V = time_dil.project(V)
        v = time_dil.project(v)

        V, v = lss.checkpoint(V, v)

        if tangent_run:
            u0, V, v, J0, G, g = tangent_segment(
                    tangent_run, u0, V, v, parameter, i, steps_per_segment,
                    simultaneous_runs)
        else:
            u0, V, v, J0, G, g = run_segment(
                    run, u0, V, v, parameter, i, steps_per_segment,
                    epsilon, simultaneous_runs)
        J_hist.append(J0)
        G_lss.append(G)
        g_lss.append(g)

        checkpoint = Checkpoint(
                u0, V, v, lss, G_lss, g_lss, J_hist, G_dil, g_dil)
        if checkpoint_path and (i) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, checkpoint)

        G = lss_gradient(checkpoint)
        print(G); sys.stdout.flush()

    if return_checkpoint:
        return checkpoint
    else:
        return np.array(J_hist).mean((0,1)), G

def shadowing(
        run, u0, parameter, subspace_dimension, num_segments,
        steps_per_segment, runup_steps, epsilon=1E-6,
        checkpoint_path=None, checkpoint_interval=1, simultaneous_runs=None,
        tangent_run=None, run_ddt=None, return_checkpoint=False):
    '''
    run: a function in the form
         u1, J = run(u0, parameter, steps, run_id)

         inputs  - u0:           init solution, a flat numpy array of doubles.
                   parameter:    design parameter, a single number.
                   steps:        number of time steps, an int.
                   run_id:       a unique identifier, a string,
                                 e.g., "segment02_init_perturb003".
         outputs - u1:           final solution, a flat numpy array of doubles,
                                 must be of the same size as u0.
                   J:            quantities of interest, a numpy array of shape
                                 (steps, n_qoi), where n_qoi is an arbitrary
                                 but consistent number, # quantities of interest.
    returns: (J, G)
        J: Time-averaged objective function, array of length n_qoi.
        G: Derivative of time-averaged objective function, array of length n_qoi
    '''
    run = RunWrapper(run)
    if tangent_run:
        tangent_run = TangentWrapper(tangent_run)
    if runup_steps > 0:
        u0, _ = run(u0, parameter, runup_steps, 'runup')

    V, v = tangent_initial_condition(subspace_dimension)
    lss = LssTangent(subspace_dimension)
    checkpoint = Checkpoint(u0, V, v, lss, [], [], [], [], [])
    if checkpoint_path:
        save_checkpoint(checkpoint_path, checkpoint)
    return continue_shadowing(
            run, parameter, checkpoint,
            num_segments, steps_per_segment, epsilon,
            checkpoint_path, checkpoint_interval,
            simultaneous_runs, tangent_run, run_ddt, return_checkpoint)
