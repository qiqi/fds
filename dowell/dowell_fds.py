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
from fds.checkpoint import *
import dowell


def solve(u, s, nsteps, run_id=None, lock=None):
    tmp_path = tempfile.mkdtemp()
    
    n_grid = u.shape[0] # number of grid points
    dt = 0.001 #time step

    out = u.copy()
    J = zeros(nsteps)
    dowell.c_run_primal(out, s, J, nsteps, n_grid, dt)
    shutil.rmtree(tmp_path)
    return out, J

iplot = 0
def save_plot():
    plot_path = os.path.join(my_path, 'dowell_plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    global iplot
    iplot += 1
    axis([-0.1,1.1,-1.0,1.0])
    xlabel('Design parameter')
    ylabel('Objective function')
    savefig(os.path.join(plot_path, str(iplot)))

n_modes = 7
k_segments = 200
n_steps = 100
n_runup = 50000

s = array([-7.0 * 9.869604401089358,150.0,0.3,0.1,0.0])

u0 = random.rand(8)
#u0 = zeros(8)
#u0[0] = 0.01
J, G = shadowing(solve, u0, s, n_modes, k_segments, n_steps, n_runup,
                 checkpoint_path='.',checkpoint_interval=20)

cp = load_last_checkpoint('.', n_modes)
verify_checkpoint(cp)

L = cp.lss.lyapunov_exponents()
print(L.shape)

# compute lyapunov exponents
#lya = np.cumsum(L,axis=1)
lya = np.zeros(L.shape)
n = np.arange(1,len(L)+1)
for i in range(L.shape[1]):
    lya[:,i] = np.cumsum(L[:,i])/n

# compute covariant vector angles
v_magnitude, sin_angle = cp.lss.lyapunov_covariant_magnitude_and_sin_angle()

min_angle = sin_angle
for i in range(n_modes):
     min_angle[i,i] = 1.0

min_angle = min_angle.min(axis=0)
print(min_angle.shape)
min_angle = min_angle.min(axis=0)
print(min_angle.shape)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(lya)
plt.savefig('lyapunov.png')

plt.figure()
plt.semilogy(sin_angle[0,1])
plt.semilogy(sin_angle[0,2])
plt.semilogy(sin_angle[1,2])
plt.legend(['01','02','12'])
plt.savefig('angles.png')

plt.figure()

plt.semilogy(min_angle)
plt.savefig('min_angle.png')

plt.show()


'''
s = linspace(0.0, 1.0, 11)
J, G = [], []
for si in s:
    u0 = zeros(127)
    u0[64] = 0.1
    Ji, Gi = shadowing(solve, u0, si, n_modes, k_segments, n_steps, n_runup)
    J.append(Ji)
    G.append(Gi)

J, G = array(J, float), array(G, float)
plot(s, J, 'o')
save_plot()

ds = 0.25
for i in range(J.shape[1]):
    plot([s-ds, s+ds], [J[:,i]-G[:,i]*ds, J[:,i]+G[:,i]*ds], '-r')
save_plot()
'''
'''
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
'''
