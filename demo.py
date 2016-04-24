from pylab import *
from numpy import *
from subprocess import check_output

iplot = 0
def save_plot():
    global iplot
    iplot += 1
    axis([27,35,65,110])
    xlabel('Design parameter')
    ylabel('Objective function')
    savefig(str(iplot))

s = linspace(28, 34, 21)

J, G = [], []
for si in s:
    out = check_output(['/usr/bin/python', 'fds.py', '--parameter', str(si-28)])
    Ji, Gi = out.strip().splitlines()
    J.append(Ji)
    G.append(Gi)

J, G = array(J, float), array(G, float)
plot(s, J, 'ok')
save_plot()

ds = 0.25
plot([s-ds, s+ds], [J-G*ds, J+G*ds], '-r')
save_plot()

# twice as long
J3 = []
for si in s:
    out = check_output(['/usr/bin/python', 'fds.py', '--parameter', str(si-28),
                        '--num_segments', '1', '--steps_per_segment', '15000'])
    Ji, _ = out.strip().splitlines()
    J3.append(Ji)

J3 = array(J3, float)
clf()
plot(s, J3, 'ok')
save_plot()

# 200 times as long
J2000 = []
for si in s:
    out = check_output(['/usr/bin/python', 'fds.py', '--parameter', str(si-28),
                        '--num_segments', '1',
                        '--steps_per_segment', '10000000'])
    Ji, _ = out.strip().splitlines()
    J2000.append(Ji)

J2000 = array(J2000, float)
clf()
plot(s, J2000, 'ok')
save_plot()

fd = (J2000[1:] - J2000[:-1]) / (s[1:] - s[:-1])
clf()
plot((s[1:] + s[:-1]) / 2, fd, 's')
plot(s, G, 'o')
xlim([27,35])
xlabel('Design parameter')
ylabel('Derivative of objective function')
legend(['Conventional finite difference 10000 time units',
        'Shadowing finite difference 15 time units'])
savefig('5')
