import os
import sys
import shutil
import tempfile
from subprocess import *

import matplotlib
matplotlib.use('Agg')
from pylab import *
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import *

solver_path = os.path.join(my_path, '..', 'tests', 'solvers', 'circular')
solver = os.path.join(solver_path, 'solver')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

def solve(u, s, nsteps, run_id=None, lock=None):
    tmp_path = tempfile.mkdtemp()
    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray(s, dtype='>d').tobytes())
    call([solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'objective.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    shutil.rmtree(tmp_path)
    return out, J[:,newaxis]

iplot = 0
def save_plot():
    plot_path = os.path.join(my_path, 'circular_plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    global iplot
    iplot += 1
    axis([-1,7,0,8])
    xlabel('Design parameter')
    ylabel('Objective function')
    savefig(os.path.join(plot_path, str(iplot)))

s = linspace(0, 6, 21)

J, G = [], []
for si in s:
    Ji, Gi = shadowing(solve, u0, si, 1, 5, 5000, 10000)
    J.append(Ji)
    G.append(Gi)

J, G = array(J, float), array(G, float)
plot(s, J, 'ok')
save_plot()

ds = 0.25
plot([s-ds, s+ds], [J[:,0]-G[:,0]*ds, J[:,0]+G[:,0]*ds], '-r')
save_plot()

for T in [50000]:
    J2000 = []
    for si in s:
        u, _ = solve(u0, si, 5000)
        _, Ji = solve(u, si, T)
        J2000.append(Ji.mean())

    J2000 = array(J2000, float)
    clf()
    plot(s, J2000, 'ok')
    save_plot()

fd = (J2000[1:] - J2000[:-1]) / (s[1:] - s[:-1])
clf()
plot((s[1:] + s[:-1]) / 2, fd, 's')
plot(s, G, 'o')
xlim([-1,7])
xlabel('Design parameter')
ylabel('Derivative of objective function')
legend(['Conventional finite difference {0} time units'.format(T/1000),
        'Shadowing finite difference 10 time units'], loc='lower right')
savefig(os.path.join(my_path, 'circular_plots', '0'))
