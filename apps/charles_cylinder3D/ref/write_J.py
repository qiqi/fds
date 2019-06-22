# This file reads in the cylinderforce.txt
# computes the Lift(t), Drag(t), RMS_Lift(t), BasePressure(t)
# all with dimensions.
# output to the files
from numpy import *
# from pylab import *

c = loadtxt("cylinderforce.txt")
c = c.reshape([-1,5])
c = c.T
c = c[2:]
myfile = open('C.txt','w')
P = 101325.

# plot Lift(t) 
# fig = figure()
# plot(c[1])
# savefig("Lift coefficient")
# close(fig)

# plot Drag(t)
# fig = figure()
# plot(c[0])
# savefig("Drag coefficient")
# close(fig)

# compute averaged C_D
myfile.write("The averaged C_D and its 2-sigma interval:\n{0}\n".format(mean(c[0])))

# compute the base pressure coefficient from line+time average
cbp2 = loadtxt("./SOLUT_4/baselinepressure.P")
cbp2 = cbp2.reshape([-1,27])
cbp2 = cbp2.T
cbp2 = cbp2[3:]

cbp2 = -(cbp2 - P)
cbp2 = cbp2.mean(axis=0)
myfile.write("C_bp average on line+time:\n{0}\n".format(mean(cbp2)))

# fig = figure()
# plot(cbp2)
# savefig("base pressure coefficient averaged over line.png")
# close(fig)

# compute averaged Lift
myfile.write("L:\n{0}\n".format(mean(c[1])))

# compute L**2
L2 = c[1]**2
myfile.write("Lift**2:\n{0}\n".format(mean(L2)))

# write to file: Lift, Drag, Square of Lift, Base pressure
savetxt('J_history.txt', (c[0], cbp2, c[1], c[1]**2))
