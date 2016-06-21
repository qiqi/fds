import sys, os
from numpy import *
from scipy.integrate import odeint
sys.path.append('/home/qiqi/git/fds')
import fds

def dudt(u):
    x, dxdt = u
    ddxdtt = (1 - x**2) * dxdt - x
    #ddxdtt = (1 - x**2 - dxdt**2) * dxdt - x
    return array([dxdt, ddxdtt])

def run(u, dt, nsteps):
    J = empty(nsteps)
    for i in range(nsteps):
        # u += dt * dudt(u)
        u = odeint(lambda u, dt : dudt(u), u, [0,dt])[1]
        J[i] = u[0]**2
    return u, J

# J, G = fds.shadowing(run, [1,1], 0.1, 1, 50, 60, 100, checkpoint_path='.', epsilon=1E-8)
# 
# u0, V, v, lss, G_lss, g_lss, J, G_dil, g_dil = fds.checkpoint.load_last_checkpoint('.', 1)
# alpha = lss.solve()
# grad_lss = (alpha[:,:,newaxis] * array(G_lss)).sum(1) + array(g_lss)
# 
# J = array(J)
# dJ = J.mean((0,1)) - J[:,-1]
# steps_per_segment = J.shape[1]
# dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
# grad_dil = dil[:,newaxis] * dJ
# 
# plot(grad_lss + grad_dil, 'o-')
# 
J, G = fds.shadowing(run, [1,1], 0.1, 1, 50, 100, 100, checkpoint_path='.', epsilon=1E-8, run_ddt=lambda u, dt: dudt(u) * dt)

checkpoint = fds.checkpoint.load_last_checkpoint('.', 1)
u0, V, v, lss, G_lss, g_lss, J, G_dil, g_dil = checkpoint
alpha = lss.solve()
grad_lss = (alpha[:,:,newaxis] * array(G_lss)).sum(1) + array(g_lss)

J = array(J)
#dJ = J.mean((0,1)) - J[:,-1]
dJ = fds.segment.trapez_mean(J,1) - J[:,-1]
steps_per_segment = J.shape[1] - 1
dil = ((alpha * G_dil).sum(1) + g_dil) / steps_per_segment
grad_dil = dil[:,newaxis] * dJ

plot(grad_lss + grad_dil, 'o-')

grad_hist= []
for i in range(1, lss.K_segments() + 1):
    grad_hist.append(fds.core.lss_gradient(checkpoint, i))
figure()
loglog(fabs(grad_hist), '-o')
grid()
plot([1,100], [1,0.01])

figure()
plot(squeeze(J).T)
