from pylab import *
from numpy import *
from subprocess import check_output

iplot = 0
def save_plot():
    global iplot
    iplot += 1
    axis([-1,7,65,110])
    xlabel('Design parameter')
    ylabel('Objective function')
    savefig(str(iplot))

s = linspace(0, 6, 21)

J, G = [], []
for si in s:
    out = check_output(['/usr/bin/python', 'fds.py', '--parameter', str(si)])
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
J2 = []
for si in s:
    out = check_output(['/usr/bin/python', 'fds.py', '--parameter', str(si),
                        '--num_segments', '1', '--steps_per_segment', '10000'])
    Ji, _ = out.strip().splitlines()
    J2.append(Ji)

J2 = array(J2, float)
clf()
plot(s, J2, 'ok')
save_plot()

# 200 times as long
J2000 = []
for si in s:
    out = check_output(['/usr/bin/python', 'fds.py', '--parameter', str(si),
                        '--num_segments', '1', '--steps_per_segment', '10000000'])
    Ji, _ = out.strip().splitlines()
    J2000.append(Ji)

J2000 = array(J2000, float)
clf()
plot(s, J2000, 'ok')
save_plot()
