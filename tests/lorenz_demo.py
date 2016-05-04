import os
import sys
import shutil
import tempfile
from subprocess import *

from pylab import *
from numpy import *

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(my_path, '..'))

from fds import finite_difference_shadowing

solver_path = os.path.join(my_path, '..', 'solvers', 'lorenz')
solver = os.path.join(solver_path, 'solver')
u0 = loadtxt(os.path.join(solver_path, 'u0'))

def solve(u, s, nsteps):
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
    return out, J

iplot = 0
def save_plot():
    plot_path = os.path.join(my_path, 'lorenz_plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    global iplot
    iplot += 1
    axis([27,35,65,110])
    xlabel('Design parameter')
    ylabel('Objective function')
    savefig(os.path.join(plot_path, str(iplot)))

s = linspace(28, 34, 21)

J, G = [], []
for si in s:
    Ji, Gi = finite_difference_shadowing(
            solve, u0, si-28, 1, 10, 1000, 5000)
    J.append(Ji)
    G.append(Gi)

J, G = array(J, float), array(G, float)
plot(s, J, 'ok')
save_plot()

ds = 0.25
plot([s-ds, s+ds], [J-G*ds, J+G*ds], '-r')
save_plot()

# twice as long
J3 = []
for si in s:
    u, _ = solve(u0, si-28, 5000)
    _, Ji = solve(u, si-28, 15000)
    J3.append(Ji.mean())

J3 = array(J3, float)
clf()
plot(s, J3, 'ok')
save_plot()

for T in [50000, 500000, 5000000]:
    J2000 = []
    for si in s:
        u, _ = solve(u0, si-28, 5000)
        _, Ji = solve(u, si-28, T)
        J2000.append(Ji.mean())

    J2000 = array(J2000, float)
    clf()
    plot(s, J2000, 'ok')
    save_plot()

fd = (J2000[1:] - J2000[:-1]) / (s[1:] - s[:-1])
clf()
plot((s[1:] + s[:-1]) / 2, fd, 's')
plot(s, G, 'o')
xlim([27,35])
xlabel('Design parameter')
ylabel('Derivative of objective function')
legend(['Conventional finite difference {0} time units'.format(T/1000),
        'Shadowing finite difference 10 time units'])
savefig(os.path.join(my_path, 'lorenz_plots', '0'))
