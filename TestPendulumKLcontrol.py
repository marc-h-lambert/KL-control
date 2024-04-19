###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# Test Pendulum with Variational control                                          #
###################################################################################

from ControlLibrary.StochasticDynamicsExample import Pendulum
from ControlLibrary.KLcontrol import KLcontrol

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
        ax[3].plot(T[0:],loss, color=col,linestyle=ls,linewidth='3')
        ax[3].set_ylabel("Cumulative loss",weight='bold')

def plotStatsTraj(ax,trajMC, controlsMC, valueMC):
    T=np.linspace(0, dt * N, N)
    Tplus = np.linspace(0, dt * N, N+1)
    # ----------- Plot Traj ------------
    meanTraj = np.mean(trajMC, axis=2)[:,0]*180/np.pi
    stdTraj = np.sqrt(np.var(trajMC, axis=2))[:,0]*180/np.pi
    ax[0].plot(Tplus, meanTraj, linewidth='3',label='mean',color='r')
    ax[0].fill(np.append(Tplus, Tplus[::-1]), np.append(meanTraj+stdTraj, (meanTraj-stdTraj)[::-1]), 'darkgray')
    ax[0].set_ylabel(r"pendulum angle $\theta^\circ$",weight='bold')

    # ----------- Plot Control ------------
    meanControl = np.mean(controlsMC, axis=1)
    stdControl = np.sqrt(np.var(controlsMC, axis=1))

    ax[1].plot(T, meanControl, linewidth='3', label='mean', color='r')
    ax[1].fill(np.append(T, T[::-1]), np.append(meanControl + stdControl, (meanControl - stdControl)[::-1]), 'darkgray')
    ax[1].set_ylabel(r"control ($\circ/s^2$)", weight='bold')

    # ----------- Plot Value ------------
    meanValue = np.mean(valueMC[0:-1],axis=1)
    stdValue = np.sqrt(np.var(valueMC[0:-1], axis=1))
    ax[2].plot(T, meanValue, linewidth='3',label='mean',color='r')
    ax[2].fill(np.append(T, T[::-1]), np.append(meanValue+stdValue, (meanValue-stdValue)[::-1]), 'darkgray')
    ax[2].set_ylabel("cost-to-go", weight='bold')
    ax[2].set_xlabel("time (s)",weight='bold')


###########################################################################################################
# MAIN PROGRAM
###########################################################################################################
if __name__ == "__main__":
    TEST=["TestSymbolicCalculus"]
    num=0
    seed=10
    dt=0.01
    T=10
    np.random.seed(seed)

    ########## PENDULUM PARAMETERS ############
    deg = math.pi / 180
    m = 1  # mass of pendulum
    l = 1  # length of pendulum
    xi = 1.  # damping
    theta0 = np.pi / 6
    dtheta0 = 0
    thetaF = 0
    dthetaF = 0
    xT=np.array([thetaF,dthetaF])

    ########## LOSS ############
    # INIT LOSS
    r = 0.01
    q = 0.01
    Q = np.identity(2) * q
    R = np.identity(1) * r
    B = np.array([[1 / (m * l * l)]])  # control matrix
    L = np.identity(1)


    if "TestPendulumPassive" in TEST:
        print("---------- Test Pendulum--------")
        ##### Test class ####
        eta = 1 * deg * 1 * deg  # np.pi / 500
        pend = Pendulum(m, l, xi, theta0, dtheta0, eta, dt, invertedPendulum=True)
        pend.propagate(T,addNoise=True)
        fig, ax = plt.subplots(2, 1, figsize=(4, 6))
        ax[0].plot(pend.trajArr()[:,1]*180/np.pi, linewidth='3', label='mean', color='r')
        ax[1].plot(pend.trajArr()[:, 1] * 180 / np.pi,pend.trajectory()[:, 2] * 180 / np.pi, linewidth='3', label='mean', color='r')
        plt.legend()
        fig.suptitle("Inverted pendulum  \n with Brownian noise on acceleration")
        plt.tight_layout()
        plt.savefig("TestPendulum.pdf", format="pdf")

    if "TestPendulumKL" in TEST:
        print("---------- Test Pendulum--------")
        eps_kl = np.array([1 * deg, 4 * deg, 6 * deg])
        eps_kl = eps_kl * eps_kl
        eta = 1 * deg * 1 * deg
        p = 0.01
        VT = np.identity(2) * p
        ##### Test class ####
        C = np.identity(1)*eta
        fig, ax = plt.subplots(4, 1, figsize=(4, 6))
        colors = np.array(["b", "g", "r"])
        styles = np.array(["-", "--", "-."])
        for i in range(0, eps_kl.shape[0]):
            eps = eps_kl[i]
            col = colors[i]
            ls = styles[i]
            pend = Pendulum(m, l, xi, theta0, dtheta0, eta, dt, invertedPendulum=True)
            controller = KLcontrol(pend, B, Q, R, VT, xT, T, eps)
            listK, listV = controller.synthetizeController()
            pend.setPolicy(controller.optimalPolicy)
            pend.propagate(T, addNoise=True)
            loss = controller.lossHisto()
            cost = controller.costTogoHisto()
            plotTrajControl(ax,pend.trajArr(),pend.controlArr(),listK,loss[:,1],idx=1,label=r'$\sqrt{\varepsilon}=$'+'{:,.2f}'.format(math.sqrt(eps)),ls=ls,col=col)
        ax[0].legend()
        plt.legend()
        fig.suptitle("Inverted pendulum  \n with Brownian noise on acceleration")
        plt.tight_layout()
        plt.savefig("TestPendulum2.pdf", format="pdf")

    if "TestPendulumKL2" in TEST:
        eta = 6 * deg * 6 * deg
        epsKL = 6 * deg * 6 * deg
        np.random.seed(seed)
        # INIT Final Cost
        p = 10
        VT = np.identity(2) * p
        C = np.identity(1) * eta
        # Test KL
        fig, ax = plt.subplots(3, 1, figsize=(4, 6))
        Nmc=30
        N=int(T/dt)
        trajMC = np.zeros([N+1, 2,Nmc])
        lossMC = np.zeros([N+1, Nmc])
        controlsMC = np.zeros([N,Nmc])
        valueMC = np.zeros([N+1,Nmc])
        for k in range(0,Nmc):
            print("MC run n°{}".format(k))
            pend = Pendulum(m, l, xi, theta0, dtheta0, eta, dt, invertedPendulum=True)
            controller = KLcontrol(pend, B, Q, R, VT, xT, T, epsKL,randomControl=True)
            listK, listV = controller.synthetizeController()
            pend.setPolicy(controller.optimalPolicy)
            pend.propagate(T, addNoise=True)
            loss = controller.lossHisto()
            cost = controller.qValueHisto()
            trajMC[:,:,k]=pend.trajArr()[:,1:]
            controlsMC[:,k]=pend.controlArr()[:,1:].reshape(-1,)
            lossMC[:,k]=loss[:,1:].reshape(-1,)
            valueMC[:,k]=cost[:,1:].reshape(-1,)
        plotStatsTraj(ax,trajMC,controlsMC, valueMC)
        ax[0].legend()

        FileName = "Test2_KL"
        ax[0].set_title("random Variational control",weight='bold')
        plt.tight_layout()
        plt.savefig(FileName + ".pdf", format="pdf")

    if "TestSymbolicCalculus" in TEST:
        eta = 6 * deg * 6 * deg
        pend = Pendulum(m, l, xi, theta0, dtheta0, eta, dt, invertedPendulum=True)
        pend.initializeAuto()
        B = np.array([[1 / (m * l * l)]])  # control matrix
        L = np.identity(1)
        C = np.identity(1) * eta
        p = 10
        VT = np.identity(2) * p
        epsilon = 6 * deg * 6 * deg
        controller = KLcontrol(pend, B, Q, R, VT, xT, T, epsilon)
        K=np.array([[1,2]])
        P=np.identity(2)
        x=np.array([7,32])
        print("Jacobian method1=", pend.jacobianTransition(x))
        print("Jacobian method2=", pend.jacobianTransitionAuto(x))
        print("Hessian method1=", pend.hessianTransition(x))
        print("Hessian method2=", pend.hessianTransitionAuto(x))
        print("Hmatrix method1=",pend.HmatrixValidation(x,K,P,dt))
        print("Hmatrix method2=", controller.HmatrixAuto(x, K, P))

plt.show()


