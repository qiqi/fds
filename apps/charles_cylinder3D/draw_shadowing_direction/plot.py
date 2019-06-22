# this file loads the checkpoints in fds
# computes djds v.s. # segments and plot

from numpy import *
import sys
sys.path.append('/scratch/niangxiu/fds/')
import fds
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')

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
start_seg = 100
start_seg_4compute = 200
N_homo = 30
K = 400

# compute djds
try:
    c, djds, interval, C1, K = pickle.load(open("change_u_finer_reso.p", "rb"))
    
except FileNotFoundError:
    print('pickle file not found, read from checkpoint files.')

    # compute djds v.s. # segment
    cp = fds.checkpoint.load_last_checkpoint('charles', N_homo)
    c = zeros([K, N_objectives])
    for i in range(start_seg, K):
        c[i] = fds.lss_gradient(cp,[start_seg,i+1])
    savetxt('djds_segment.txt', c)

    # for djds
    djds = zeros(N_objectives)
    interval = zeros(N_objectives)
    C1 = zeros(N_objectives)
    for i in range(N_objectives):
        djds[i], interval[i], C1[i] = compute(c[:,i], start_seg_4compute)

    pickle.dump((c, djds, interval, C1, K), open("change_u_finer_reso.p", "wb"))


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


# normalize
def normalize(djds, interval, c, C1):
    djds[0] /= F0/U0
    interval[0] /= F0/U0
    c[:,0] /= F0/U0
    C1[0] /= F0/U0 * (t0/DT)**0.5

    djds[1] /= P0/U0
    interval[1] /= P0/U0
    c[:,1] /= P0/U0 
    C1[1] /= P0/U0 * (t0/DT)**0.5

    djds[2] /= F0/r0
    interval[2] /= F0/r0
    c[:,2] /= F0/r0
    C1[2] /= F0/r0 * (t0/DT)**0.5

    djds[3] /= F0**2/r0
    interval[3] /= F0**2/r0
    c[:,3] /= F0**2/r0
    C1[3] /=  F0**2/r0 * (t0/DT)**0.5


xx = arange(start_seg_4compute,K)*1.0
xx /= t0/DT
normalize(djds, interval, c, C1) 


with open("result.txt",'w') as f:
    print('djds', djds, file = f)
    print('interval, +-', interval, file = f)


c2, djds2, interval2, C1mid, Kmid = pickle.load(open("../change_u_mid_resolution/change_u_mid_reso.p", "rb"))
normalize( djds2, interval2, c2, C1mid )

# plot history of two djds
yl = ['$[d\langle D_r \\rangle / dU] \,/\, [F_0/U_0]$', '$[d\langle S_b \\rangle / dU] \,/\, [P_0/U_0]$','','']
for i in range(4):
    fig = plt.figure(figsize=(8,6))
    # gcf().subplots_adjust(bottom=0.15)
    plt.plot(xx, c[start_seg_4compute:,i], label='finer mesh', linewidth=4, linestyle='solid', color='k')
    plt.plot(xx, djds[i]+C1[i]*xx**-0.5, 'k--')
    plt.plot(xx, djds[i]-C1[i]*xx**-0.5, 'k--')
    plt.plot(xx, c2[start_seg_4compute:,i], label='coarser mesh', linewidth=4, linestyle='dashdot', color='k')
    plt.plot(xx, djds2[i]+C1mid[i]*xx**-0.5, 'k--')
    plt.plot(xx, djds2[i]-C1mid[i]*xx**-0.5, 'k--')
    plt.grid
    plt.legend(loc=3)
    plt.xlabel('$T/t_0$')
    plt.ylabel(yl[i])
    plt.ylim([0.5* max(c[start_seg_4compute:,i]), 1.2* max(c[start_seg_4compute:,i])])
    plt.ticklabel_format(style='sci', axis='y', scilimits=(3,3))
    plt.tight_layout()
    plt.savefig('dJds_history' + str(i) + suffix[i])
    plt.close(fig)

