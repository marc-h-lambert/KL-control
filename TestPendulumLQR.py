###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# Test Pendulum                                                                   #
###################################################################################

from ControlLibrary.StochasticDynamicsExample import Pendulum
from ControlLibrary.LQR import LQR

import math
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################################
# PLOT METHODS
###########################################################################################################

def plotTrajControl(ax,traj,controls,listK,loss,idx=1,label='',col='r',ls='-'):
    T=traj[:, 0]
    print("'lentraj=",len(T))
    if label=='':
        ax[0].plot(T, traj[:, idx] * 180 / np.pi, color=col, linestyle=ls, linewidth='3')
    else:
        ax[0].plot(T, traj[:, idx] * 180 / np.pi, label=label,color=col, linestyle=ls, linewidth='3')
    ax[0].set_ylabel(r"pendulum angle $\theta^\circ$", weight='bold')

    ax[1].plot(T[0:-1], controls[:,1],color=col,linestyle=ls,linewidth='3')
    ax[1].set_ylabel("controls", weight='bold')

    ax[2].plot(T[0:-1], np.array(listK)[:,0,0],color=col,linestyle=ls,linewidth='3')
    ax[2].plot(T[0:-1], np.array(listK)[:,0,1],color=col,linestyle=ls,linewidth='3')
    ax[2].set_ylabel("Gains", weight='bold')

    if not loss.shape ==0:
        ax[3].plot(loss, color=col,linestyle=ls,linewidth='3')
        ax[3].set_ylabel("Cumulative loss",weight='bold')


###########################################################################################################
# MAIN PROGRAM
###########################################################################################################
if __name__ == "__main__":
    TEST=["TestPendulumLQR"]
    num=0
    seed=10
    dt=0.1
    T=10
    np.random.seed(seed)

    ########## PENDULUM PARAMETERS ############
    m = 1  # mass of pendulum
    l = 1  # length of pendulum
    xi = 1.  # damping
    deg = math.pi / 180
    theta0 = np.pi / 6
    dtheta0 = 0
    thetaF = 0
    dthetaF = 0
    eta = 1 * deg * 1 * deg  # np.pi / 500

    ########## LOSS ############
    # INIT LOSS
    r = 0.01
    q = 0.01
    p = 0.01
    Q = np.identity(2) * q
    R = np.identity(1) * r
    VT = np.identity(2) * p

    if "TestPendulumPassive" in TEST:
        print("---------- Test Pendulum--------")
        ##### Test class ####
        pend = Pendulum(m, l, xi, theta0, dtheta0, eta, dt,invertedPendulum=True)
        pend.propagate(T,addNoise=True)
        num = num + 1
        plt.figure(num)
        fig, ax = plt.subplots(2, 1, figsize=(4, 6))
        ax[0].plot(pend.trajArr()[:,1]*180/np.pi, linewidth='3', label='mean', color='r')
        ax[1].plot(pend.trajArr()[:, 1] * 180 / np.pi,pend.trajArr()[:, 2] * 180 / np.pi, linewidth='3', label='mean', color='r')
        plt.legend()
        fig.suptitle("Inverted pendulum  \n with Brownian noise on acceleration")
        plt.tight_layout()
        plt.savefig("TestPendulum.pdf", format="pdf")

    if "TestPendulumLQR" in TEST:
        print("---------- Regulation of the inverted Pendulum around 0 (vertical) --------")
        zerosIsUp = True
        pend = Pendulum(m, l, xi, theta0, dtheta0, eta, dt, zerosIsUp)
        Jf = pend.jacobianDynamic(np.array([thetaF,dthetaF]))
        B = np.array([[1 / (m * l * l)]])   # control matrix
        L = np.identity(1)
        C = np.identity(1)*eta
        controller=LQR(Jf,B,Q,R,VT,T,L,C,[theta0],[dtheta0],[thetaF],[dthetaF],pend)
        listK, listV=controller.synthetizeController()
        pend.setPolicy(controller.optimalPolicy)
        pend.propagate(T,addNoise=True)
        print("TRAJ1=", pend.trajArr())
        loss=controller.lossHisto()
        cost=controller.costTogoHisto()
        num = num + 1
        plt.figure(num)
        fig, ax = plt.subplots(4, 1, figsize=(4, 6))
        plotTrajControl(ax,pend.trajArr(),pend.controlArr(),listK,loss[:,1],idx=1)
        plt.legend()
        fig.suptitle("Inverted pendulum  \n with Brownian noise on acceleration")
        plt.tight_layout()
        plt.savefig("TestPendulum.pdf", format="pdf")

plt.show()


