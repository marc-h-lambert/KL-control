

###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# LQR controller                                                                  #
###################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

from Core.StochasticController import StochasticController
from Core.StochasticDynamicalSystem import LinearSystem

plt.rcParams['pdf.fonttype'] = 42

class LQR(StochasticController):
    # LQR with discrete finite horizon setting
    # dp = F.(x,p).T+Bp.(x,p).T + L.w
    # dx = pdt
    # A is supposed to be of the form [[0,Id],F]
    # B is supposed to be of the form [[0],Bp]
    def __init__(self,F,Bp,Q,R,VT,T,L,C,x0,p0,xT,pT,system=[]):
        # setting the linear system (in continuous time)
        d,p=Bp.shape
        dt=system.dt
        if system==[]:
            system=LinearSystem(x0,p0,L,C,F,Bp)
        else:
            # specify the system in case of linearized LQR to compute the exact loss
            system=system
        self.xT=xT
        self.pT=pT
        self.Xref=np.concatenate((xT, pT), axis=0)
        # setting the LQR (in discrete time)
        M=np.concatenate((np.zeros([d,d]),np.identity(d)),axis=1)
        self.A=np.identity(2*d)+np.concatenate((M,F),axis=0)*dt
        self.B=np.concatenate((np.zeros([d,p]),Bp),axis=0)*dt
        self.Q=Q*dt
        self.R=R*dt
        self.VT=VT
        transitionCost=lambda X,u,t: (X-self.Xref).T.dot(self.Q.dot(X-self.Xref))+u.T.dot(self.R.dot(u))
        finalCost = lambda X: (X-self.Xref).T.dot(self.VT.dot(X-self.Xref))
        super().__init__(system, p, transitionCost, finalCost, T)

    @staticmethod
    def gainLQR(A, V, R, B):
        S = R + B.T.dot(V).dot(B)
        return LA.inv(S).dot(B.T).dot(V).dot(A)

    # backward Riccati in discrete finite horizon setting (DARE=Discrete Algebraic Riccati Equation)
    @staticmethod
    def backwardDARE(A, V, R, B, Q):
        S = R + B.T.dot(V).dot(B)
        return Q - A.T.dot(V).dot(B).dot(LA.inv(S)).dot(B.T).dot(V).dot(A) + A.T.dot(V).dot(A)

    #@overrides
    def synthetizeController(self):
        N=int(self.cost.finalTime/self.dt)
        n=2*self.system.d
        listK = np.zeros([N, self.dimControl,n])
        listV = np.zeros([N + 1, n, n])
        listV[N] = self.VT
        V = self.VT
        print("A=", self.A)
        print("B=", self.B)
        print("Q=", self.Q)
        print("R=", self.R)
        print("PT=", V)
        for i in range(N, 0, -1):
            # print("--- iteration N°", i)
            K = LQR.gainLQR(self.A, V, self.R, self.B)
            print(K)
            V = LQR.backwardDARE(self.A, V, self.R, self.B, self.Q)
            listK[i - 1] = K
            listV[i - 1] = V
        self.listK=listK
        self.listV=listV
        super().synthetizeController()
        print('lenK=', len(listK))
        return listK, listV

    #@overrides
    def optimalPolicy(self,state,time):
        t=int(time/self.dt)
        K=self.listK[t]
        return -K.dot(state-self.Xref).reshape(self.dimControl,1)


