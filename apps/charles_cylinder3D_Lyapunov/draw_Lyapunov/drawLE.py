# this file loads the checkpoints in fds
# compute and plot all LEs and their converging history

from numpy import *
import sys
sys.path.append('/scratch/niangxiu/fds_4CLV_finer_reso/')
import fds
import matplotlib
matplotlib.use('Agg')
import pickle
from matplotlib.pyplot import *
rcParams.update({'axes.labelsize':'xx-large'})
rcParams.update({'xtick.labelsize':'xx-large'})
rcParams.update({'ytick.labelsize':'xx-large'})
rcParams.update({'legend.fontsize':'xx-large'})
rc('font', family='sans-serif')


# compute the confidence interval of a converging sequence a[start_seg:],
# the converging rate is assumed to be C1 * n^-0.5
# return the estimated converging value, confidence interval, and C1
def compute(d,start_seg):
    assert len(d.shape) == 1
    n_partition = 1001
    delta = 0.001 * absolute(d).max()
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


N_objectives = 4
start_seg = 100 # we look at only segments 5,6,7,8,...
start_seg_4compute = 500
N_homo = 40


try:
    LE, err, L, K = pickle.load(open("LE.p", "rb"))

except FileNotFoundError:
    print('pickle file not found, read from checkpoint files.')
    cp = fds.checkpoint.load_last_checkpoint('charles', N_homo)
    K = size(cp.g_dil)

    # for LE
    L = cp.lss.lyapunov_exponents() 
    L = fds.timeseries.exp_cum_mean(L)
    print(L.shape)
    LE  = zeros(N_homo)
    err = zeros(N_homo)
    for i in range(N_homo):
        LE[i], err[i], _ = compute(L[:,i], start_seg_4compute)
    pickle.dump((LE, err,L, K), open("LE.p", "wb"))


# normalize
suffix = ['_U_D', '_U_Pb', '', '']
U0 = 33
D = 0.25e-3
Z = 2*D
rho = 1.18
F0 = 0.5 * rho * U0**2 * D * Z
P0 = 0.5 * rho * U0**2
r0 = U0/D
t0 = D/U0
DT = 1e-8 * 200


xx = arange(start_seg,K)*1.0
xx /= t0/DT
LE /= DT/t0 
L /= DT/t0 
err /= DT/t0


# plot all LE values and confidence interval
fig = figure(figsize = (8,6))
I = argsort(LE)[::-1]
print(LE[I[0]])
errorbar(arange(N_homo)+1, LE[I], yerr = err[I],fmt = 'o') 
plot([0,18],[0,0],'--')
gcf().subplots_adjust(left=0.15)
xlabel('number of LE')
ylabel('LE$\,/\, t_0^{-1}$')
tight_layout()
savefig('LE_err.png')
close(fig)


# plot convergence history for LE
fig = figure(figsize = (8,6)) 
gcf().subplots_adjust(left=0.15)
plot(xx, L[start_seg:])
grid()
xlabel('$T\,/\, t_0$')
ylabel('LE$\,/\, t_0^{-1}$')
tight_layout()
savefig('LE_history.png')
close(fig)


f = open('result.txt', 'w')
print('LE', LE[I], file = f)
print('interval, +-',err[I], file = f)
print('sum of all LEs', sum(LE), file=f)
print('sum of all positive LEs', sum(np.maximum(LE,0)), file=f)
