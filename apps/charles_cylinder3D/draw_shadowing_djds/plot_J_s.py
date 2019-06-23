# this file reads in the C.txt in each folder and draws the objective-parameter plot, with uncertanty in the objectives
from numpy import *
import os
import pickle
import matplotlib
matplotlib.use('Agg')
from pylab import *

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')


def plotall(prmt, obj, n, ds, obj_err, djds, djds_err, filename, xl, yl, prmtmid, objmid, nmid, obj_errmid, djdsmid, djds_errmid):
    print('plotting...')
    fig = figure(figsize=(8,6))
    
    # plot(prmt, obj, 'ko', markersize=10)
    errorbar(prmt, obj, yerr= obj_err, fmt='ko', capthick=2, capsize=8, elinewidth=2, markersize=10, ecolor='k', mfc='k', mec='k', label='finer mesh')
    for j in range(-10,11):
        dJ = (j/10.0 * djds_err  + djds)*ds
        plot([prmt[n]-ds,prmt[n]+ds], [obj[n] - dJ,obj[n] + dJ], 'k-')

    errorbar(prmtmid, objmid, yerr= obj_errmid, fmt='r>', capthick=2, capsize=8, elinewidth=2, markersize=11, ecolor='r', mfc='r', mec='r', label='coarser mesh')
    for j in range(-10,11):
        dJ = (j/10.0 * djds_errmid + djdsmid)*ds
        plot([prmtmid[nmid]-ds,prmtmid[nmid]+ds], [objmid[nmid] - dJ,objmid[nmid] + dJ], 'r-')

    xlabel(xl)
    ylabel(yl)
    legend()
    tight_layout()
    savefig(filename)
    close()


U = array([])
drag = array([])
drag_err= array([])
bp = array([])
bp_err = array([])
djds        = array([ 2.38917569e+00,  2.17401134e+00,  -5.95867716e+00,   1.67171742e-01])
djds_err    = array([ 9.63829140e-02,  2.07996383e-01,   2.32644009e-01,   3.08744268e-01])


round_arr = array([-6e3, 0, 2e3, 4e3, 6e3])
# w_arr = round_arr * 2 * pi
l = array([])
l_err = array([])
l2 = array([])
l2_err = array([])


# read data
try:
    U, drag, drag_err, bp, bp_err, l, l_err, l2, l2_err, djds, djds_err, round_arr = pickle.load(open("J_s_dJds.p", "rb"))

except FileNotFoundError:
    print("pickle file not found, read from folders.")
    pwd_arr = ['U30','U32','U33_backup','U34','U36']
    for pwd in pwd_arr:
        with open(os.path.join(pwd, "charles.in"), "r") as f:
            for line in f:
                cc = line.split()
                if size(cc) != 0:
                    if cc[0] == "left":
                        U = hstack((U, float(cc[3])))
        with open(os.path.join(pwd, "C.txt"), "r") as f:
            lines = f.readlines()
            drag = hstack((drag, float((lines[1].split())[0])))
            drag_err = hstack((drag_err, float((lines[1].split())[1])))
            bp = hstack((bp, float((lines[3].split())[0])))
            bp_err = hstack((bp_err, float((lines[3].split())[1])))

    pwd_arr = array(['rotate_m6e3', 'U33_backup','rotate_2e3_CFL', 'rotate_4e3','rotate_6e3'])
    for pwd in pwd_arr:
        with open(pwd+"/C.txt", "r") as f:
            lines = f.readlines()
            l = hstack((l, float((lines[5].split())[0])))
            l_err = hstack((l_err, float((lines[5].split())[1])))
            l2 = hstack((l2, float((lines[7].split())[0])))
            l2_err = hstack((l2_err, float((lines[7].split())[1])))

    # nomalization
    U0 = 33
    D = 0.25e-3
    Z = 2*D
    rho = 1.18
    F0 = 0.5 * rho * U0**2 * D * Z
    P0 = 0.5 * rho * U0**2
    r0 = U0/D

    U /= U0
    round_arr /= r0

    drag /= F0
    drag_err /= F0

    print(bp)
    bp /= P0
    bp_err /= P0
    print(bp)

    l /= F0
    l_err /= F0

    l2 /= F0**2
    l2_err /= F0**2

    pickle.dump((U, drag, drag_err, bp, bp_err, l, l_err, l2, l2_err, djds, djds_err, round_arr), open("J_s_dJds.p", "wb"))


Umid, dragmid, drag_errmid, bpmid, bp_errmid, lmid, l_errmid, l2mid, l2_errmid, djdsmid, djds_errmid, round_arrmid =\
        pickle.load(open("../mid_reso/finite_difference/J_s_dJds_mid.p", "rb"))


plotall(U, drag, 2, 0.02, drag_err, djds[0], djds_err[0], 'U_drag.png', '$U / U_0$', '$\langle D_r \\rangle/F_0$', 
        Umid, dragmid, 3, drag_errmid, djdsmid[0], drag_errmid[0])
plotall(U, bp,   2, 0.02, bp_err,   djds[1], djds_err[1], 'U_bp.png',   '$U/U_0$', '$\langle S_b \\rangle/P_0$',
        Umid, bpmid, 3, bp_errmid, djdsmid[1], djds_errmid[1])
plotall(round_arr, l,  1, 0.01, l_err,  djds[2], djds_err[2], 'w_l.png',  '$\omega/\omega_0$', '$\langle L \\rangle/F_0$',
        round_arrmid, lmid, 2, l_errmid,  djdsmid[2], djds_errmid[2])
plotall(round_arr, l2, 1, 0.01, l2_err, djds[3], djds_err[3], 'w_l2.png', '$\omega/\omega_0$', '$\langle L^2 \\rangle/F_0^2$',
        round_arrmid, l2mid, 2, l2_errmid, djdsmid[3], djds_errmid[3])

with open("result.txt",'w') as myfile:
    print(drag, bp, l, l2, file = myfile)
