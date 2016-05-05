import os
import sys
import shutil
import tempfile
from subprocess import *
from multiprocessing.pool import ThreadPool

import matplotlib
matplotlib.use('Agg')
from pylab import *
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import finite_difference_shadowing

solver_path = os.path.join(my_path, '..', '..', 'les-inwd', 'apps')
py_script = os.path.join(solver_path, 'jet.py')

random.seed(0)
u0 = load(os.path.join(solver_path, 'jet_initial_state.npy'))
u0 = ravel(u0)

def solve(u0, jet_V, nsteps, run_id):
    Re = 1000
    base_path = os.path.join(my_path, 'flow3d')
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    solver_path = os.path.join(base_path, run_id)
    if not os.path.exists(solver_path):
        os.mkdir(solver_path)
        u0 = u0.reshape([16, 80, 20, 20])
        save(os.path.join(solver_path, 'initial_state.npy'), u0)
        savetxt(os.path.join(solver_path, 'parameters.txt'), array([Re, jet_V]))
        call(['python', py_script, str(Re), str(jet_V), str(int(nsteps))],
             cwd=solver_path, stdout=open(os.devnull, 'w'))
    u1 = load(os.path.join(solver_path, 'final_state.npy'))
    J = loadtxt(os.path.join(solver_path, 'jet_quantities.txt'))
    J = J.reshape([-1,2])
    J[:,1] = J[:,1]**2 / 2
    return ravel(u1), J

iplot = 0
def save_plot(s, J):
    plot_path = os.path.join(my_path, 'flow3d_plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    global iplot
    iplot += 1
    for i in range(J.shape[1]):
        subplot(J.shape[1],1,i+1)
        plot(s, J[:,i])
        axis([0,2,0,0.1])
        xlabel('Design parameter')
        ylabel('Objective function {0}'.format(i+1))
    savefig(os.path.join(plot_path, str(iplot)))

if False:
    s = linspace(0.5, 1.5, 32)

    def run(si):
        u, _ = solve(u0, si, 200)
        _, Ji = solve(u, si, 500)
        return Ji

    threads = ThreadPool()
    res = []
    for si in s:
        res.append(threads.apply_async(run, (si,)))
    J_fd = []
    for i in range(len(res)):
        J_fd.append(res[i].get())

    J_fd = array(J_fd, float)
    clf()
    save_plot(s, J_fd.mean(1))
    save('J_fd.npy', J_fd)

Ji, Gi = finite_difference_shadowing(
         solve, u0, 1.0, 14, 20, 100, 0, eps=1E-4)
