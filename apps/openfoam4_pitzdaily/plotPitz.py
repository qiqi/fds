# This files reads in 'pitz.txt', which contains djds v.s. # segments
# computes the coefficient of the T^-0.5 convergence rate
# plot the convegence history on a log-log scale, as well as the error cone
from numpy import *
from matplotlib.pyplot import *

def compute(d):
    C = zeros([201, 4])
    C1 = zeros(4)
    djds = zeros(4)
    for i in range(4):
        for j in range(201):
            temp = ones (d.shape[0])
            for n in range (5, d.shape[0]):
                temp[n] = abs(((d[-1,i] + 0.0003*(j-100)) - d[n,i]) * n**0.5)
            C[j,i] = max(temp[5:])
        C1[i] = min(C[:,i])
        djds[i] = d[-1,i] + 0.0003 *(argmin(C[:,i],axis=0)-100)
        interval = C1*d.shape[0]**-0.5 
    return djds, interval, C1

# for s = 10
d = loadtxt('pitz.txt')
for i in range(4):
    d[:,i] = d[:,i] * 0.1 / 10**(2**i)
djds, interval,C1 = compute(d)
print('djds', djds)
print('interval, +-', interval)

# for s = 11
d = loadtxt('pitz11.txt')
for i in range(4):
    d[:,i] = d[:,i] * 0.1 / 10**(2**i)
djds, interval,C1 = compute(d)
print('djds11', djds)
print('interval11, +-', interval)

for i in range(4):
    fig = figure(figsize=(6,4))
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    gcf().subplots_adjust(bottom=0.15)
    loglog(arange(5,200)+1, abs(d[5:,i] - djds[i]), 'o-')
    loglog([6, d.shape[0]], [C1[i]*6**-0.5, C1[i]*d.shape[0]**-0.5])
    grid
    xlabel('Time segment: i')
    ylabel(r"$\left| (dJ/ds)_0 - (dJ/ds)_i \right|$")
    ylim([1e-4, 1e-1])
    savefig('pitzdaily_dJds_hitory' + str(i))
