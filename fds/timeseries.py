from numpy import *

def mean_std(x):
    n = len(x)
    i = array(linspace(0, n, 9), int)
    x_i = array([x[i0:i1].mean(0) for i0, i1 in zip(i[:-1], i[1:])])
    return x.mean(0), x_i.std(0) / sqrt(len(x_i))

def windowed_mean(a):
    win = sin(linspace(0, pi, a.shape[0]+2)[1:-1])**2
    return (a * win[:,newaxis]).sum(0) / win.sum()

def exp_cum_mean(x):
    x, x_mean = array(x), []
    for n in range(1, len(x) + 1):
        w = 1 - exp(-arange(1,n+1) / sqrt(n))
        x = array(x)
        w = w.reshape([-1] + [1] * (x.ndim - 1))
        x_mean.append((x[:n] * w).sum(0) / w.sum())
    return array(x_mean)
