
###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# Variational controller                                                          #
###################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import sympy as sym

from Core.StochasticController import StochasticController
from LQR import LQR
from Core.StochasticDynamicalSystem import LinearSystem

plt.rcParams['pdf.fonttype'] = 42

class KLcontrol(StochasticController):
    # LQR with discrete finite horizon setting
    # dp = f(x,p).T+Bp.(x,p).T + L.w
    # Set autoGrad=True if you want to use the symbolic calculus to compute the
    # gradient and Hessian (may be long...)
    # otherwise the system must implements the methods at hand
    def __init__(self,system,Bp,Q,R,VT,xT,T,epsilon,randomControl=False,autoGrad=False):
        d, p = Bp.shape
        dt=system.dt
        if autoGrad:
            system.initializeAuto()
        self.B=np.concatenate((np.zeros([d,p]),Bp),axis=0)*dt
        self.Q=Q*dt
        self.R=R*dt
        self.VT=VT
        self.xT=xT
        self.eps=epsilon
        self.randomControl=randomControl
        self.autoGrad=autoGrad
        transitionCost=lambda X,u,t: X.T.dot(self.Q.dot(X))+u.T.dot(self.R.dot(u))
        finalCost = lambda X: X.T.dot(self.VT.dot(X))
        super().__init__(system, p, transitionCost, finalCost, T)

    # Jacobian of the transition
    def Jf(self, X):
        X = X.reshape([2 * self.system.d, ])
        if self.autoGrad:
            return self.system.jacobianTransitionAuto(X)
        else:
            return self.system.jacobianTransition(X)

    # Matrix H which depends on the Hessian of the transition
    def Hmatrix(self,X, K,P):
        X = X.reshape([2 * self.system.d,])
        d = self.system.d
        u=-K.dot(X).reshape(-1,1)
        if self.autoGrad:
            dF = self.system.hessianTransitionAuto(X)
            Xkk = self.system.discreteTransitionAuto(X) + self.B.dot(u) * self.dt
            v = P.dot(Xkk).reshape(2, 1)
            res = np.tensordot(dF, v, axes=([0], [0]))
            res = res.reshape(2 * d, 2 * d)
            return res
        else:
            dF = self.system.hessianTransition(X)
            Xkk = self.system.discreteTransition(X) + self.B.dot(u) * self.dt
            v = P.dot(Xkk).reshape(2, 1)
            res = np.tensordot(dF, v, axes=([0], [0]))
            res = res.reshape(2 * d, 2 * d)
            return np.float64(res)

    @staticmethod
    def fmeanCKF(f, mu, sqrtP, *args):
        mu = mu.reshape(-1, 1)
        d = sqrtP.shape[0]
        fmean = 0 * f(mu, *args)

        for i in range(0, d):
            vi = sqrtP[:, i].reshape(d, 1)
            sigmaPointi_Plus = mu + vi * math.sqrt(d)
            sigmaPointi_Moins = mu - vi * math.sqrt(d)
            Wi = 1 / (2 * d)
            fmean = fmean + Wi * f(sigmaPointi_Plus, *args) + Wi * f(sigmaPointi_Moins, *args)
        return fmean

    @staticmethod
    def JPJ(x, Jf, P, *args):
        return Jf(x, *args).T.dot(P).dot(Jf(x, *args))

    def cov(self,P):
        S = self.R + self.B.T.dot(P).dot(self.B)
        return LA.inv(S)*self.eps

    def gainKL(self,xk, Pk, Pkk):
        S = self.R + self.B.T.dot(Pkk).dot(self.B)
        # compute the expectation with cubature points
        EA = KLcontrol.fmeanCKF(self.Jf, xk, LA.cholesky(LA.inv(Pk) * self.eps))
        return LA.inv(S).dot(self.B.T).dot(Pkk).dot(EA)

    # backward Riccati in discrete finite horizon setting (DARE=Discrete Algebraic Riccati Equation)
    def iterateKL(self,xk,Pk, Pkk):
        covV = LA.inv(Pk) * self.eps
        sqrtV = LA.cholesky(covV)
        S = self.R + self.B.T.dot(Pkk).dot(self.B)
        A = KLcontrol.fmeanCKF(self.Jf, xk, sqrtV)
        M1 = KLcontrol.fmeanCKF(KLcontrol.JPJ, xk, sqrtV, self.Jf, Pkk)
        K = self.gainKL(xk, Pk, Pkk)
        M2 = KLcontrol.fmeanCKF(self.Hmatrix, xk, sqrtV, K, Pkk)
        #M2 = KLcontrol.fmeanCKF(self.system.Hmatrix, xk, sqrtV, K, Pkk, self.dt)
        return self.Q - A.T.dot(Pkk).dot(self.B).dot(LA.inv(S)).dot(self.B.T).dot(Pkk).dot(A) + M1 + M2

    #@overrides
    def synthetizeController(self):
        N=int(self.cost.finalTime/self.dt)
        n=2*self.system.d
        listK = np.zeros([N, self.dimControl,n])
        listV = np.zeros([N + 1, n, n])
        listV[N] = self.VT
        Pkk = self.VT
        for i in range(N, 0, -1):
            print("--- iteration N°", i)
            ## Inner loop to find Pk (we start with LQR guess)
            Pk=LQR.backwardDARE(self.Jf(self.xT), Pkk, self.R, self.B, self.Q)
            for k in range(0, 10):
                Pk = self.iterateKL(self.xT,Pk, Pkk) #xT is fixed here
            K = self.gainKL(self.xT, Pk, Pkk)
            Pkk = Pk
            listK[i - 1] = K
            listV[i - 1] = Pk
        self.listK=listK
        self.listV=listV
        super().synthetizeController()
        return listK, listV

    #@overrides
    def optimalPolicy(self,state,time):
        t=int(time/self.dt)
        K = self.listK[t]
        P = self.listV[t]
        mean_u=-K.dot(state).reshape(self.dimControl,)
        if self.randomControl:
            u=np.random.multivariate_normal(mean_u, self.cov(P), 1)
        else:
            u=mean_u
        return u.reshape(self.dimControl,1)

    def qValueHisto(self):
        self.system.reinitialize()
        currtraj, currControls = self.system.propagate(self.dt, self.cost.finalTime, addNoise=True)
        N = np.shape(currtraj)[0]
        qvalue = np.zeros([N, 2])
        for i in range(0,N):
            time=currtraj[i,0]
            x=currtraj[i,1:]
            P=self.listV[i]
            (sign, logdetinvP) = np.linalg.slogdet(self.eps * LA.inv(P))
            logq = -0.5 * x.T.dot(P).dot(x) / self.eps - 0.5 * logdetinvP - math.log(2 * math.pi)
            qvalue[i, 0] = time
            qvalue[i,1] = -self.eps  * logq
        return qvalue







