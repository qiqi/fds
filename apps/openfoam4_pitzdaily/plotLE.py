# this file load checkpoints in fds
# it generates the convergence history of all Lyapunov exponent v.s. #segments
# computes the error bounds of all LEs
# plots all LEs sorted by their values

from numpy import *
from matplotlib.pyplot import *
import sys
sys.path.append('/home/ubuntu/data/git/fds/')
import fds

nLE = 16
nstart = 100
cp = fds.checkpoint.load_last_checkpoint('pitzdaily', nLE)
cp.lss.lyapunov_exponents()
L = cp.lss.lyapunov_exponents()
L = L[:200]
L = fds.timeseries.exp_cum_mean(L)

ntrial = 401
eps = 0.0003
C = zeros([ntrial, nLE])
C1 = zeros(nLE)
LE = zeros(nLE)
for i in range(nLE):
    for j in range(ntrial):
        temp = ones (L.shape[0])
        for n in range (nstart, L.shape[0]):
            temp[n] = abs(((L[-1,i] + eps*(j-(ntrial-1)/2)) - L[n,i]) * n**0.5)
        C[j,i] = max(temp[nstart:])
    C1[i] = min(C[:,i])
    LE[i] = L[-1,i] + eps *(argmin(C[:,i],axis=0)-(ntrial-1)/2)

err = C1*L.shape[0]**-0.5
print('LE', LE)
print('interval, +-',err)

# for i in range(11,16):
    # fig = figure(figsize=(6,4))
    # gcf().subplots_adjust(bottom=0.15)
    # loglog(arange(nstart,200)+1, abs(L[nstart:,i] - LE[i]), 'o-')
    # loglog([nstart, L.shape[0]], [C1[i]*nstart**-0.5, C1[i]*L.shape[0]**-0.5])
    # grid
    # xlabel('Time segments')
    # ylabel('dJ/ds')
    # savefig('LE_converge' + str(i))

fig1 = figure(figsize = (6,5)) 
plot(arange(20,200)+1, L[20:200])
grid()
xlabel('Time segment')
ylabel('Lyapunov exponents')
savefig('LE_history.png')

fig2 = figure(figsize = (6,5))
I = argsort(LE)[::-1]
errorbar(arange(16)+1, LE[I], yerr = err[I],fmt = 'o') 
plot([0,18],[0,0],'--')
xlim ([0,18])
gcf().subplots_adjust(left=0.15)
xlabel('n-th largest Lyapunov exponent')
ylabel('value of Lyapunov exponent')
savefig('LE_err.png')
