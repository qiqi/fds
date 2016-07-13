from dowell_fds import *

iplot = 0
def save_plot():
    plot_path = os.path.join(my_path, 'dowell_plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    global iplot
    iplot += 1
    #axis([-0.1,1.1,-1.0,1.0])
    xlabel('Design parameter')
    ylabel('Objective function')
    savefig(os.path.join(plot_path, str(iplot)))


n_modes = 3
k_segments = 10
n_steps = 1000
n_runup = 50000
filt = 10.0

s = linspace(-8.0, -6.0, 11) * 9.869604401089358

J, G = [], []
Jf, Gf =[], []
for si in s:
    print("Starting " + str(si))
    u0 = zeros(8)
    u0[0] = 0.01
    Ji, Gi = shadowing(solve, u0, si, n_modes, k_segments, n_steps, n_runup, epsilon=1e-6,filt=0.0)
    J.append(Ji)
    G.append(Gi)
    print("Starting " + str(si) + " with filt")
    Jfi, Gfi = shadowing(solve, u0, si, n_modes, k_segments, n_steps, n_runup, epsilon=1e-6,filt=filt)
    Jf.append(Jfi)
    Gf.append(Gfi)
    print("Finished " + str(si))

J, G = array(J, float), array(G, float)
Jf, Gf = array(Jf, float), array(Gf, float)

plot(s, J, 'o')
save_plot()

ds = 1.5
for i in range(J.shape[1]):
    plot([s-ds, s+ds], [J[:,i]-G[:,i]*ds, J[:,i]+G[:,i]*ds], '-r')
    plot([s-ds, s+ds], [Jf[:,i]-Gf[:,i]*ds, Jf[:,i]+Gf[:,i]*ds], '-b')
save_plot()





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
savefig(os.path.join(my_path, 'dowell_plots', '0'))
'''
