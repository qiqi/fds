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

solver_path = os.path.join(my_path, '..', 'tests', 'solvers', 'lorenz')
solver = os.path.join(solver_path, 'solver')
u0 = loadtxt(os.path.join(solver_path, 'u0'))
serial_mode = False

def get_host_dir(run_id):
    return os.path.join('lorenz_demo', run_id)

if not serial_mode:
    os.makedirs(get_host_dir(''))
    import h5py
    def save_hdf5(arr, path):
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('field', data=arr)
        return
    def load_hdf5(path):
        with h5py.File(path, 'r') as handle:
            field = handle['/field'][:].copy()
        return field
    path = get_host_dir('u0')
    save_hdf5(u0, path)
    u0 = path

def solve(u, s, nsteps, run_id=None, lock=None):
    print u, run_id

    if serial_mode:
        tmp_path = tempfile.mkdtemp()
    else:
        u = load_hdf5(u)
        tmp_path = get_host_dir(run_id)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

    with open(os.path.join(tmp_path, 'input.bin'), 'wb') as f:
        f.write(asarray(u, dtype='>d').tobytes())
    with open(os.path.join(tmp_path, 'param.bin'), 'wb') as f:
        f.write(asarray(s, dtype='>d').tobytes())
    call([solver, str(int(nsteps))], cwd=tmp_path)
    with open(os.path.join(tmp_path, 'output.bin'), 'rb') as f:
        out = frombuffer(f.read(), dtype='>d')
    with open(os.path.join(tmp_path, 'objective.bin'), 'rb') as f:
        J = frombuffer(f.read(), dtype='>d')
    J = transpose([J, 100 * ones(J.size)])

    if serial_mode:
        shutil.rmtree(tmp_path)
    else:
        tmp_output = os.path.join(tmp_path, 'output.h5')
        save_hdf5(out, tmp_output)
        out = tmp_output

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
    Ji, Gi = shadowing(solve, u0, si-28, 1, 10, 1000, 5000, get_host_dir=get_host_dir)
    J.append(Ji)
    G.append(Gi)

J, G = array(J, float), array(G, float)
plot(s, J, 'o')
save_plot()

ds = 0.25
for i in range(J.shape[1]):
    plot([s-ds, s+ds], [J[:,i]-G[:,i]*ds, J[:,i]+G[:,i]*ds], '-r')
save_plot()

# twice as long
J3 = []
for si in s:
    u, _ = solve(u0, si-28, 5000)
    _, Ji = solve(u, si-28, 15000)
    J3.append(Ji.mean(0))

J3 = array(J3, float)
clf()
plot(s, J3, 'o')
save_plot()

for T in [50000, 500000, 5000000]:
    J2000 = []
    for si in s:
        u, _ = solve(u0, si-28, 5000)
        _, Ji = solve(u, si-28, T)
        J2000.append(Ji.mean(0))

    J2000 = array(J2000, float)
    clf()
    plot(s, J2000, 'o')
    save_plot()

fd = (J2000[1:] - J2000[:-1]) / (s[1:] - s[:-1])[:,newaxis]
clf()
plot((s[1:] + s[:-1]) / 2, fd, 's')
plot(s, G, 'o')
xlim([27,35])
xlabel('Design parameter')
ylabel('Derivative of objective function')
legend(['Conventional finite difference {0} time units'.format(T/1000), '',
        'Shadowing finite difference 10 time units', ''])
savefig(os.path.join(my_path, 'lorenz_plots', '0'))
