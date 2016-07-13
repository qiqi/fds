from dowell_fds import *

n_modes = 7
k_segments = 20
n_steps = 1000
n_runup = 50000

s = -9.0 * 9.869604401089358
#u0 = random.rand(8)
u0 = zeros(8)
u0[0] = 0.01
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


