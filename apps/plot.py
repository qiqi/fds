# this file loads the checkpoints in fds
# computes djds v.s. # segments and plot

from numpy import *
import sys
sys.path.append('/scratch/niangxiu/fds/')
import fds
from matplotlib.pyplot import *

N_objectives = 4
start_seg = 5 # we look at only segments 5,6,7,8,...
N_homo = 20

# compute djds v.s. # segment
cp = fds.checkpoint.load_last_checkpoint('charles', N_homo)
K = size(cp.g_dil)
c = zeros([K, N_objectives])
for i in range(start_seg, K):
    c[i] = fds.lss_gradient(cp,[start_seg,i+1])
savetxt('djds_segment.txt', c)

# compute the confidence interval of a converging sequence a[start_seg:],
# the convenging rate is assumed to be C1 * n^-0.5
# return the estimated converging value, confidence interval, and C1
def compute(d,start_seg):
    assert len(d.shape) == 1
    n_partition = 1001
    delta = 0.0000001 * d.max()
    C = zeros(n_partition)
    for j in range(n_partition):
        temp = zeros (d.shape[0])
        for n in range (start_seg, d.shape[0]):
            temp[n] = abs(((d[-1] + delta*(j-n_partition/2.0)) - d[n]) * (n+1) **0.5)
        C[j] = max(temp) 
    C1 = min(C)
    limit = d[-1] + delta* (argmin(C)-n_partition/2.0)
    interval = C1*(d.shape[0]+1)**-0.5 
    return limit, interval, C1

# for djds
djds = zeros(N_objectives)
interval = zeros(N_objectives)
C1 = zeros(N_objectives)
for i in range(N_objectives):
    djds[i], interval[i], C1[i] = compute(c[:,i], start_seg)
print('djds', djds)
print('interval, +-', interval)

# plot convergence history of error of djds
for i in range(4):
    fig = figure(figsize=(6,4))
    gcf().subplots_adjust(bottom=0.15)
    loglog(arange(start_seg,K)+1, abs(c[start_seg:,i] - djds[i]), 'o-')
    loglog([start_seg+1, K+1], [C1[i]*(start_seg+1)**-0.5, C1[i]*(K+1)**-0.5])
    grid
    xlabel('Time segment: i')
    ylabel(r"$\left| (dJ/ds)_0 - (dJ/ds)_i \right|$")
    savefig('dJds_error_history' + str(i))
    close(fig)
    
# plot history of djds
for i in range(4):
    fig = figure(figsize=(6,4))
    gcf().subplots_adjust(bottom=0.15)
    plot(c[:,i])
    grid
    xlabel('Time segment: i')
    ylabel('dj/ds')
    savefig('dJds_history' + str(i))
    close(fig)

# for LE
L = cp.lss.lyapunov_exponents()
L = fds.timeseries.exp_cum_mean(L)

LE  = zeros(N_homo)
err = zeros(N_homo)
C1  = zeros(N_homo)
for i in range(N_homo):
    LE[i], err[i], C1[i] = compute(L[:,i], start_seg)
print('LE', LE)
print('interval, +-',err)

# plot convergence history for LE
fig = figure(figsize = (6,5)) 
plot(L)
grid()
xlabel('Time segment')
ylabel('Lyapunov exponents')
savefig('LE_history.png')
close(fig)

# plot all LE values and confidence interval
fig = figure(figsize = (6,5))
I = argsort(LE)[::-1]
errorbar(arange(N_homo)+1, LE[I], yerr = err[I],fmt = 'o') 
plot([0,18],[0,0],'--')
gcf().subplots_adjust(left=0.15)
xlabel('n-th largest Lyapunov exponent')
ylabel('value of Lyapunov exponent')
savefig('LE_err.png')
close(fig)
