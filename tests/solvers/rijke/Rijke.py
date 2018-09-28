from __future__ import division

import os

from pylab import *
from numpy import *
from scipy.integrate import odeint
from scipy.sparse import csr_matrix

#from numba import jit

#import progressbar

__all__ = ['Rijke_advect', 'Rijke_advect_integrate', 'Param', 'Scheme',
           'Rijke_advect_tanadj', 'Rijke_advect_tanadj_v2',
           'fourier2real', 'cheb_grid']

def cheb_grid(n):
    'Chebyshev grid in [-1,1] of n+1 points'
    return -cos(pi*arange(n+1,dtype=float)/n)

def cheb_diff(n):
    'Chebyshev collocation derivative matrix of n+1 points'
    if not hasattr(cheb_diff, '_cache'):
        cheb_diff._cache = {}
    if n not in cheb_diff._cache:
        x = cheb_grid(n)
        c = hstack([2., ones(n-1), 2.]) * (-1)**arange(n+1);
        X = outer(x, ones(n+1))
        dX = X-X.T;
        D = outer(c, 1 / c) / (dX + eye(n+1))
        D -= diag(D.sum(1))
        cheb_diff._cache[n] = D
    return cheb_diff._cache[n]

def q_dot(delayed_velocity):
    '5th order polynomial fit heat release model'
    #return (sqrt(abs(1./3+delayed_velocity)) - sqrt(3))
    fifth_order_poly = [0.5,-0.108,-0.044,0.059,-0.012]
    return dot(fifth_order_poly, delayed_velocity**arange(1,6))

def q_dot_tan(delayed_velocity):
    'Derivative of 5th order polynomial fit heat release model'
    fifth_order_poly = [0.5,-0.108,-0.044,0.059,-0.012]
    return dot(fifth_order_poly, arange(1,6) * delayed_velocity**arange(5))

def Rijke_advect(u, Ncheb, tau, tauL, jpi, damping, kgM, sinjpixf, cosjpixf,
                 alpha=None):
    '''
    ddt of Rijke model.
    alpha=None: non-chaotic case.
    Otherwise alpha is the coupling coefficient between Lorenz and Rijke.
    '''
    dudt = empty_like(u)
    if alpha is not None:
        x, y, z = u[-3:]
        u = u[:-3]
        rho, beta, sigma = 28, 8./3, 10
        dxdt = 1 / tauL * (sigma * (y - x))
        dydt = 1 / tauL * (x * (rho - z) - y)
        dzdt = 1 / tauL * (x * y - beta * z)
        velocity_fluctuation = alpha * x / (rho - 1)
        dudt[-3:] = [dxdt, dydt, dzdt] #  dxyz_dt
    else:
        velocity_fluctuation = 0
        dxyz_dt = []
    velocity = u[-Ncheb:]
    u = u[:-Ncheb]
    Ng = u.size // 2
    eta, etadotonjpi = u.reshape([2,Ng])

    acoustic_velocity_at_flame = dot(eta, cosjpixf) + velocity_fluctuation
    heat_release = kgM * q_dot(velocity[-1])

    # deta_dt and detadotonjpi_dt
    dudt[:Ng] = jpi * etadotonjpi
    dudt[Ng:2*Ng] = -jpi*eta - damping*etadotonjpi - 2*sinjpixf*heat_release

    D = cheb_diff(Ncheb)
    dudt[2*Ng:2*Ng+Ncheb] = -2 / tau * (D[1:,0] * acoustic_velocity_at_flame +
                                        dot(D[1:,1:], velocity)) # dvelocity_dt
    return dudt

def Rijke_advect_tan(u, v, tau, tauL, jpi, damping, kgM, sinjpixf, cosjpixf,
                     alpha=None):
    'Tangent ddt of Rijke model'
    if alpha is not None:
        x, y, z = u[-3:]
        dx, dy, dz = v[-3:]
        u = u[:-3]
        v = v[:-3]
        rho, beta, sigma = 28, 8./3, 10
        dxdt_p = 1 / tauL * (sigma * (dy - dx))
        dydt_p = 1 / tauL * (dx * (rho - z) - x * dz - dy)
        dzdt_p = 1 / tauL * (dx * y + x * dy - beta * dz)
        velocity_fluctuation_p = alpha * dx / (rho - 1)
        dxyz_dt_p = [dxdt_p, dydt_p, dzdt_p]
    else:
        velocity_fluctuation_p = 0
        dxyz_dt_p = []
    n = u.size // 3
    eta, etadotonjpi, velocity = u.reshape([3,n])
    eta_p, etadotonjpi_p, velocity_p = v.reshape([3,n])

    acoustic_velocity_at_flame_p = dot(eta_p, cosjpixf) + velocity_fluctuation_p
    velocity_p = hstack([acoustic_velocity_at_flame_p, velocity_p])
    heat_release_p = kgM * q_dot_tan(velocity[-1]) * velocity_p[-1]

    deta_dt_p = jpi * etadotonjpi_p
    detadotonjpi_dt_p = (-jpi*eta_p - damping*etadotonjpi_p \
            - 2*sinjpixf*heat_release_p)
    dvelocity_dt_p = -2 / tau * (cheb_diff(n) * velocity_p)[1:]
    return hstack([deta_dt_p, detadotonjpi_dt_p, dvelocity_dt_p, dxyz_dt_p])

def Rijke_advect_adj(u, w, tau, tauL, jpi, damping, kgM, sinjpixf, cosjpixf,
                     alpha=None):
    'Adjoint of the ddt of Rijke model'
    if alpha is not None:
        x, y, z = u[-3:]
        dxdt_a, dydt_a, dzdt_a = w[-3:]
        u = u[:-3]
        w = w[:-3]
    n = u.size // 3
    eta, etadotonjpi, velocity = u.reshape([3,n])
    deta_dt_a, detadotonjpi_dt_a, dvelocity_dt_a = w.reshape([3,n])

    velocity_a = -2 / tau * (hstack([0, dvelocity_dt_a]) * cheb_diff(n))
    eta_a = -jpi * detadotonjpi_dt_a
    etadotonjpi_a = -damping * detadotonjpi_dt_a + jpi * deta_dt_a
    heat_release_a = -2*dot(sinjpixf, detadotonjpi_dt_a)

    velocity_a[-1] += kgM * q_dot_tan(velocity[-1]) * heat_release_a
    acoustic_velocity_at_flame_a = velocity_a[0]
    velocity_a = velocity_a[1:]
    eta_a += acoustic_velocity_at_flame_a * cosjpixf
    if alpha is not None:
        rho, beta, sigma = 28, 8./3, 10
        x_a = acoustic_velocity_at_flame_a * alpha / (rho - 1) \
            + 1 / tauL * (-sigma * dxdt_a + (rho - z) * dydt_a + y * dzdt_a)
        y_a = 1 / tauL * (sigma * dxdt_a - dydt_a + x * dzdt_a)
        z_a = 1 / tauL * (-x * dydt_a - beta * dzdt_a)
        xyz_a = [x_a, y_a, z_a]
    else:
        xyz_a = []
    return hstack([eta_a, etadotonjpi_a, velocity_a, xyz_a])

def fourier2real(param, u, x):
    'Transform u from Fourier to real at specified x locations'
    Ng       = param.Ng
    Ncheb    = param.Ncheb
    j        = arange(1,Ng+1)
    jpi      = j.reshape([Ng] + [1] * array(x).ndim)*pi
    cosjpixf = cos(jpi*x)
    sinjpixf = sin(jpi*x)
    eta = u[:,:Ng]
    etadotonjpi = u[:,Ng:2*Ng]
    v = u[:,2*Ng:2*Ng+Ncheb]
    return dot(eta, cosjpixf), dot(etadotonjpi, sinjpixf), v

def Rijke_advect_integrate(param,scheme,u0):
    Ng       = param.Ng
    Ncheb    = param.Ncheb
    j        = arange(1,Ng+1)
    jpi      = j*pi
    damping  = param.c_1*j**2+ param.c_2*j**(0.5)
    cosjpixf = cos(jpi*param.x_f)
    sinjpixf = sin(jpi*param.x_f)
    args = param.Ncheb, param.tau, param.tauL, jpi, damping, param.kgM, sinjpixf, cosjpixf
    if scheme.method == 'odeint':
        integrator = odeint
    elif scheme.method == 'wray':
        integrator = wray
    if param.alpha is not None:
        args = args + (param.alpha,)
    u = integrator(lambda u,t : Rijke_advect(u, *args), u0, scheme.t)
    eta = u[:,:Ng]
    etadotonjpi = u[:,Ng:2*Ng]
    v = u[:,2*Ng:2*Ng+Ncheb]
    up, pdot, v = fourier2real(param, u, param.x_f)
    return up, pdot, v, u[-1]

ALPHA, GAMMA, ETA = [0,0,0], [8./15, 5./12, 3./4], [0, -17./60, -5./12]

def wray(ddt_fun, u0, t, verbose=False):
    'A 3-stage Runge Kutta scheme that is relatively easy to adjoint'
    u = empty([t.size, u0.size])
    u[0,:] = u0
    N0, N1 = empty([2, u0.size])
    #if verbose:
     #   bar = progressbar.ProgressBar(maxval=t.size)
     #   bar.start()
    for i in range(1, t.size):
        if verbose and i % 1000 == 0:
            bar.update(i)
        dt = t[i] - t[i-1]
        for k in range(3):
            N0 = ddt_fun(u0, t[i] + ALPHA[k] + dt)
            u0 += dt * (GAMMA[k] * N0 + ETA[k] * N1)
            N1[:] = N0
        u[i,:] = u0
    #if verbose:
    #    bar.finish()
    return u

def wray_rk(k, ddt_fun, t, dt, u, N0, N1):
    '''
    Perform the k-th stage (0-based) of the 3-stage Runkge Kutta by Wray.
    k:       (0, 1, or 2) stage to be performed
    ddt_fun: a function to be called as ddt_fun(u, t)
    t, dt:   beginning and size of the time step (all 3 stages)
    u:       (input, output) the state to be updated
    N0, N1:  (input, output) temporary storage
    '''
    N0 = ddt_fun(u, t + ALPHA[k] + dt)
    u += dt * (GAMMA[k] * N0 + ETA[k] * N1)
    N1[:] = N0

def wray_rk_tan(k, ddt_tan, t, dt, u, v, M0, M1):
    '''
    Tangent of the k-th stage (0-based) of the 3-stage Runkge Kutta by Wray.
    k:       (0, 1, or 2) stage to be performed
    ddt_tan: tangent of ddt_fun in wray_rk, to be called as ddt_tan(u, v, t)
    t, dt:   beginning and size of the time step (all 3 stages)
    u:       (input) the state before calling the corresponding wray_rk
    v:       (input, output) the tangent state to be updated
    M0, M1:  (input, output) temporary storage for the tangent
    '''
    M0 = ddt_tan(u, v, t + ALPHA[k] + dt)
    v += dt * (GAMMA[k] * M0 + ETA[k] * M1)
    M1[:] = M0

def wray_rk_adj(k, ddt_adj, t, dt, u, w, M0_dt, M1_dt):
    '''
    Adjoint of the k-th stage (0-based) of the 3-stage Runkge Kutta by Wray.
    To be called in reverse order, k being 2, 1, and then 0.
    k:       (2, 1, or 0) stage to be performed
    ddt_adj: adjoint of ddt_fun in wray_rk, to be called as ddt_adj(u, w, t)
             dot(v, ddt_adj(u, w, t)) == dot(w, ddt_tan(u, v, t))
    t, dt:   beginning and size of the time step (all 3 stages)
    u:       (input) the state before calling the corresponding wray_rk
    w:       (input, output) the adjoint state to be updated
    M0_dt, M1_dt: (input, output) temporary storage for the adjoint
    '''
    M0_dt[:] = GAMMA[k] * w + M1_dt
    M1_dt[:] = ETA[k] * w
    w += dt * ddt_adj(u, M0_dt, t + ALPHA[k] + dt)

'''
def rescale(v):
    if (v**2).sum() > 1E100:
        v /= 1E100

def wray_tan_adj(ddt_fun, ddt_tan, ddt_adj, u0, v0, w0, t, verbose):
    u, v, w = empty([3, t.size, u0.size])
    u_tmp = empty([t.size - 1, 3, u0.size])
    u[0,:] = u0
    v[0,:] = v0
    N0, N1, M0, M1 = empty([4, u0.size])
    ALPHA, GAMMA, ETA = [0,0,0], [8./15, 5./12, 3./4], [0, -17./60, -5./12]
    if verbose:
        bar = progressbar.ProgressBar(maxval=t.size)
        bar.start()
    for i in range(1, t.size):
        if verbose and i % 1000 == 0:
            bar.update(i)
        dt = t[i] - t[i-1]
        for k in range(3):
            u_tmp[i-1, k, :] = u0
            wray_rk_tan(k, ddt_tan, t[i], dt, u0, v0, M0, M1)
            wray_rk(k, ddt_fun, t[i], dt, u0, N0, N1)
        u[i,:] = u0
        v[i,:] = v0
        rescale(v0)
    M0[:] = 0
    M1[:] = 0
    if verbose:
        bar.finish()
        bar.start()
    for i in reversed(range(1, t.size)):
        if verbose and i % 1000 == 0:
            bar.update(t.size - i)
        w[i,:] = w0
        dt = t[i] - t[i-1]
        for k in reversed(range(3)):
            u0 = u_tmp[i-1, k, :]
            wray_rk_adj(k, ddt_adj, t[i], dt, u0, w0, M0, M1)
        rescale(w0)
    if verbose:
        bar.finish()
    w[0,:] = w0
    return u, v, w
'''

class Rescaler:
    '''
    Orthonormalize a number of vectors with respect to each other periodically
    The interval of rescaling is specified in the beginning.
    The upper-triangular matrices used to orthonormalize the vectors
    are stored for reconstruction.
    '''
    def __init__(self, interval, capacity, vec_size, num_vec):
        self.interval = interval
        self.counter = 0
        self.R = []
        self.v = empty([capacity, vec_size, num_vec])

    def __call__(self, v):
        assert self.counter < self.v.shape[0]
        if self.counter % self.interval == 0:
            q, r = linalg.qr(v.T)
            v[:] = q.T
            self.R.append(r)
        self.v[self.counter] = v.T
        self.counter += 1
        if self.counter == self.v.shape[0]:
            self._reconstruct()

    def _reconstruct(self):
        R, D = eye(self.v.shape[2]), zeros(self.v.shape[2])
        self.scale = zeros([self.v.shape[0], self.v.shape[2]])
        for iR in range(len(self.R)-1, 0, -1):
            #M = dot(linalg.inv(self.R[iR]), M)
            R = linalg.solve(self.R[iR], R)
            D += log10(abs(diag(R)))
            R /= abs(diag(R))
            #for i in range(iR * self.interval, (iR + 1) * self.interval):
            i_min = (iR - 1) * self.interval
            i_max = iR * self.interval - 1
            for i in range(i_max, max(0, i_min) - 1, -1):
                self.v[i] = dot(self.v[i], R)
                vi_scale = sqrt((self.v[i]**2).sum(0))
                self.scale[i] = D + log10(vi_scale)
                self.v[i] /= vi_scale
        self.rescaled = self.v
        del self.v

def wray_tan_adj_v2(ddt_fun, ddt_tan, ddt_adj, u0, v0, w0, t, verbose=False):
    assert u0.ndim == 1 and v0.ndim == 2 and w0.ndim == 2
    assert u0.shape[0] == v0.shape[1] == w0.shape[1]
    u_tmp = empty([t.size, 3, u0.size])
    N0, N1 = empty((2, u0.size))
    M0, M1 = empty((2,) + v0.shape)
    ALPHA, GAMMA, ETA = [0,0,0], [8./15, 5./12, 3./4], [0, -17./60, -5./12]
    if verbose:
        bar = progressbar.ProgressBar(maxval=t.size)
        bar.start()
    rescale_tan = Rescaler(interval=100, capacity=t.size,
                           vec_size=v0.shape[1], num_vec=v0.shape[0])
    rescale_tan(v0)
    for i in range(1, t.size):
        if verbose and i % 1000 == 0:
            bar.update(i)
        dt = t[i] - t[i-1]
        for k in range(3):
            u_tmp[i-1, k, :] = u0
            for j in range(v0.size // u0.size):
                wray_rk_tan(k, ddt_tan, t[i], dt, u0, v0[j], M0[j], M1[j])
            wray_rk(k, ddt_fun, t[i], dt, u0, N0, N1)
        rescale_tan(v0)
    u_tmp[-1, 0] = u0
    M0, M1 = zeros((2,) + w0.shape)
    if verbose:
        bar.finish()
        bar.start()
    rescale_adj = Rescaler(interval=100, capacity=t.size,
                           vec_size=v0.shape[1], num_vec=v0.shape[0])
    rescale_adj(w0)
    for i in reversed(range(1, t.size)):
        if verbose and i % 1000 == 0:
            bar.update(t.size - i)
        dt = t[i] - t[i-1]
        for k in reversed(range(3)):
            u0 = u_tmp[i-1, k, :]
            for j in range(w0.size // u0.size):
                wray_rk_adj(k, ddt_adj, t[i], dt, u0, w0[j], M0[j], M1[j])
        rescale_adj(w0)
    if verbose:
        bar.finish()
    return u_tmp[:,0], rescale_tan, rescale_adj

def Rijke_advect_tanadj(param,scheme,u0,v0,w0,verbose=False):
    Ng       = param.Ng
    j        = arange(1,Ng+1)
    jpi      = j*pi
    damping  = param.c_1*j**2+ param.c_2*j**(0.5)
    cosjpixf = cos(jpi*param.x_f)
    sinjpixf = sin(jpi*param.x_f)
    args = param.tau, jpi, damping, param.kgM, sinjpixf, cosjpixf
    assert scheme.method == 'wray'
    if param.alpha is not None:
        args = args + (param.alpha,)
    return wray_tan_adj(lambda u,t : Rijke_advect(u, *args),
                        lambda u,v,t : Rijke_advect_tan(u, v, *args),
                        lambda u,w,t : Rijke_advect_adj(u, w, *args),
                        u0, v0, w0, scheme.t, verbose)

def Rijke_advect_tanadj_v2(param,scheme,u0,v0,w0,verbose=False):
    Ng       = param.Ng
    j        = arange(1,Ng+1)
    jpi      = j*pi
    damping  = param.c_1*j**2+ param.c_2*j**(0.5)
    cosjpixf = cos(jpi*param.x_f)
    sinjpixf = sin(jpi*param.x_f)
    args = param.tau, jpi, damping, param.kgM, sinjpixf, cosjpixf
    assert scheme.method == 'wray'
    if param.alpha is not None:
        args = args + (param.alpha,)
    return wray_tan_adj_v2(lambda u,t : Rijke_advect(u, *args),
                           lambda u,v,t : Rijke_advect_tan(u, v, *args),
                           lambda u,w,t : Rijke_advect_adj(u, w, *args),
                           u0, v0, w0, scheme.t, verbose)

class Param:
    def __init__(self, Ng, Ncheb, x_f, c_1, c_2, kgM, tau, tauL,alpha=None):
        self.Ng = Ng
        self.Ncheb = Ncheb
        self.x_f = x_f
        self.c_1 = c_1
        self.c_2 = c_2
        self.kgM = kgM
        self.tau = tau
        self.tauL = tauL
        self.alpha = alpha

class Scheme:
    def __init__(self, dt, T, method='odeint'):
        self.dt = dt
        self.T = T
        self.t = dt * arange(round(T/dt) + 1)
        self.method = method

def test_sameness(alpha=None, u0_size=1):
    param = Param(Ng=10, Ncheb=10, x_f=0.3, c_1=0.05, c_2=0.01, kgM=.9, tau=0.04,
                  alpha=alpha)
    ndim = param.Ng*3 if alpha is None else param.Ng*3+3
    u0, v0, w0 = ones([3, ndim]) * u0_size
    v0 = v0.reshape([1,-1])
    w0 = w0.reshape([1,-1])
    scheme = Scheme(0.001, 20)
    XeL,YeL,v,_ = Rijke_advect_integrate(param,scheme,u0)
    subplot(2,1,1); plot(scheme.t, XeL); plot(scheme.t, YeL)
    subplot(2,1,2); plot(scheme.t, v)
    figure()
    scheme = Scheme(0.0005, 20, 'wray')
    XeL,YeL,v,_ = Rijke_advect_integrate(param,scheme,u0.copy())
    subplot(2,1,1); plot(scheme.t, XeL); plot(scheme.t, YeL)
    subplot(2,1,2); plot(scheme.t, v)
    figure()
    u,v,w = Rijke_advect_tanadj_v2(param,scheme,u0,v0,w0,True)
    u,up,uv = fourier2real(param, u, param.x_f)
    #Ng = param.Ng
    #j = arange(1,Ng+1)
    #jpi = j*pi
    #cosjpixf = cos(jpi*param.x_f)
    subplot(2,1,1); plot(scheme.t, u)
    plot(scheme.t, up)
    subplot(2,1,2); plot(scheme.t, uv)

def test_tangent(alpha=None):
    param = Param(Ng=10, Ncheb=5, x_f=0.3, c_1=0.05, c_2=0.01, kgM=0.9, tau=0.04,
            alpha=alpha)
    scheme = Scheme(0.001, 1, 'wray')
    ndim = param.Ng*3 if alpha is None else param.Ng*3+3
    u0, v0, w0 = random.rand(3, ndim)
    v0 = v0.reshape([1,-1])
    w0 = w0.reshape([1,-1])
    u,v,w = Rijke_advect_tanadj_v2(param,scheme,u0.copy(),v0.copy(),w0.copy())
    v = (10 ** v.scale) * v.rescaled[:,:,0]
    EPS = exp(linspace(log(0.1), log(1E-8), 8))
    ERR = []
    for eps in EPS:
        up,_,_ = Rijke_advect_tanadj_v2(param,scheme,u0+v[0]*eps,v0.copy(),w0.copy())
        um,_,_ = Rijke_advect_tanadj_v2(param,scheme,u0-v[0]*eps,v0.copy(),w0.copy())
        ERR.append(linalg.norm((up - um) / (2*eps) - v))
    figure()
    loglog(EPS, ERR, '--o')
    grid()

def test_adjoint(alpha=None):
    param = Param(Ng=10, Ncheb=5, x_f=0.3, c_1=0.05, c_2=0.01, kgM=0.9, tau=0.04,
            alpha=alpha)
    scheme = Scheme(0.001, 1, 'wray')
    ndim = param.Ng*3 if alpha is None else param.Ng*3+3
    u0, v0, w0 = random.rand(3, ndim)
    v0 = v0.reshape([1,-1])
    w0 = w0.reshape([1,-1])
    u,v,w = Rijke_advect_tanadj_v2(param,scheme,u0.copy(),v0.copy(),w0.copy())
    v = v.rescaled[:,:,0] * 10**v.scale
    w = w.rescaled[::-1,:,0] * 10**w.scale[::-1]
    v_dot_w = (v * w).sum(1)
    print(v_dot_w.max(), v_dot_w.min())

def test_multi_adjoint(alpha=None):
    param = Param(Ng=10, Ncheb=5, x_f=0.3, c_1=0.05, c_2=0.01, kgM=0.9, tau=0.04,
            alpha=alpha)
    scheme = Scheme(0.001, 1, 'wray')
    ndim = param.Ng*3 if alpha is None else param.Ng*3+3
    u0 = random.rand(ndim)
    v0 = random.rand(5, ndim)
    w0 = random.rand(5, ndim)
    u,v,w = Rijke_advect_tanadj_v2(param,scheme,u0.copy(),v0.copy(),w0.copy())
    v_val = v.rescaled[:] * 10**v.scale[:,newaxis,:]
    w_val = w.rescaled[::-1] * 10**w.scale[::-1,newaxis,:]
    v_dot_w = (v_val[:,:,:,newaxis] * w_val[:,:,newaxis,:]).sum(1)
    # print(v_dot_w.max(0))
    print(v_dot_w.max(0) - v_dot_w.min(0))

if __name__ == '__main__':
    #test_sameness()
    #test_tangent()
    test_adjoint()
    test_multi_adjoint()
    #test_sameness(0.01, 1E-3)
    #test_tangent(0.01)
    #test_adjoint(0.01)
    #test_multi_adjoint(0.01)
