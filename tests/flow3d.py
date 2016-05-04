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

solver_path = os.path.join(os.path.expanduser('~'), 'git', 'les-inwd', 'apps')
py_script = os.path.join(solver_path, 'jet.py')

random.seed(0)
u0 = load(os.path.join(solver_path, 'jet_initial_state.npy'))

def solve(u0, jet_V, nsteps):
    Re = 1000
    tmp_path = tempfile.mkdtemp()
    save(os.path.join(tmp_path, 'initial_state.npy'), u0)
    call(['python', py_script, str(Re), str(jet_V), str(int(nsteps))],
         cwd=tmp_path, stdout=open(os.devnull, 'w'))
    u1 = load(os.path.join(tmp_path, 'final_state.npy'))
    J = loadtxt(os.path.join(tmp_path, 'jet_quantities.txt'))
    J[:,1] = J[:,1]**2 / 2
    shutil.rmtree(tmp_path)
    return u1, J

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
