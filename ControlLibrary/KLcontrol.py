
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
    # Xref is supposed to be an array of (T/dt) reference points
    def __init__(self,system,Bp,Q,R,VT,Xrefs,T,epsilon,randomControl=False,autoGrad=False):
        d, p = Bp.shape
        dt=system.dt
        if autoGrad:
            system.initializeAuto()
        self.B=np.concatenate((np.zeros([d,p]),Bp),axis=0)*dt
        self.Q=Q*dt
        self.R=R*dt
        self.VT=VT
        self.Xrefs=Xrefs
        self.eps=epsilon
        self.randomControl=randomControl
        self.autoGrad=autoGrad
        transitionCost=lambda X,u,t: (X-self.Xrefs[int(t/dt),:]).T.dot(self.Q.dot(X-self.Xrefs[int(t/dt),:]))+u.T.dot(self.R.dot(u))
        finalCost = lambda X: X.T.dot(self.VT.dot(X))
        super().__init__(system, p, transitionCost, finalCost, T)

    def F(self,X):
        if self.autoGrad:
            X = X.reshape([2 * self.system.d, ])
            return self.system.discreteTransitionAuto(X)
        else:
            return self.system.discreteTransition(X)

    # Jacobian of the transition
    def Jf(self, X):
        if self.autoGrad:
            X = X.reshape([2 * self.system.d, ])
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

    @staticmethod
    def JPF(x, Jf, F, B, feedback, alphak, alphakk, Pkk, betak, Kk, *args):
        return Jf(x).T.dot(Pkk).dot(F(x)+B.dot(feedback(x,alphak, betak,Kk))-alphakk)

    def cov(self,P):
        S = self.R + self.B.T.dot(P).dot(self.B)
        return LA.inv(S)*self.eps

    def gain(self,alphak, Pk, Pkk):
        S = self.R + self.B.T.dot(Pkk).dot(self.B)
        M=LA.inv(S).dot(self.B.T).dot(Pkk)
        # compute the expectation with cubature points
        EJf = KLcontrol.fmeanCKF(self.Jf, alphak, LA.cholesky(LA.inv(Pk) * self.eps))
        return M.dot(EJf)

    def bias(self,alphak, Pk, alphakk, Pkk):
        S = self.R + self.B.T.dot(Pkk).dot(self.B)
        M=LA.inv(S).dot(self.B.T).dot(Pkk)
        # compute the expectation with cubature points
        Ef = KLcontrol.fmeanCKF(self.F, alphak, LA.cholesky(LA.inv(Pk) * self.eps))
        return -1*M.dot(Ef-alphakk)

    def meanFeedback(self,xk, alphak, betak,Kk):
        return betak-Kk.dot(xk-alphak).reshape(self.dimControl,1)

    # variational backward propagation of the covariance Pk
    def variationalBackwardRiccati(self,alphak, Pk, Pkk, Kk):
        covV = LA.inv(Pk) * self.eps
        sqrtV = LA.cholesky(covV)
        S = self.R + self.B.T.dot(Pkk).dot(self.B)
        A = KLcontrol.fmeanCKF(self.Jf, alphak, sqrtV)
        M1 = KLcontrol.fmeanCKF(KLcontrol.JPJ, alphak, sqrtV, self.Jf, Pkk)
        M2 = KLcontrol.fmeanCKF(self.Hmatrix, alphak, sqrtV, Kk, Pkk)
        #M2 = KLcontrol.fmeanCKF(self.system.Hmatrix, xk, sqrtV, K, Pkk, self.dt)
        return self.Q - A.T.dot(Pkk).dot(self.B).dot(LA.inv(S)).dot(self.B.T).dot(Pkk).dot(A) + M1 + M2

    # variational backward propagation of the mean alphak
    def variationalBackwardMean(self, k, alphak, Pk, alphakk, Pkk,betak,Kk):
        Ejpf = KLcontrol.fmeanCKF(KLcontrol.JPF, alphak, LA.cholesky(LA.inv(Pk) * self.eps),self.Jf, self.F, self.B,\
                self.meanFeedback,alphak,alphakk, Pkk, betak, Kk)
        alpha=self.Xrefs[k].reshape(-1,1)-LA.inv(self.Q).dot(Ejpf)
        return alpha

    #@overrides
    def synthetizeController(self):
        N=int(self.cost.finalTime/self.dt)
        n=2*self.system.d
        self.listK = np.zeros([N, self.dimControl,n])
        self.listalpha = np.zeros([N, n,1])
        self.listbeta= np.zeros([N, self.dimControl,1])
        self.listV = np.zeros([N + 1, n, n])
        self.listV[N] = self.VT
        Pkk = self.VT
        alphakk=self.Xrefs[-1,:].reshape(n,1)
        print(alphakk)
        for k in range(N, 0, -1):
            print("--- iteration N°", k)
            alphak=alphakk
            Pk=Pkk
            #Pk=LQR.backwardDARE(self.Jf(self.xT), Pkk, self.R, self.B, self.Q)
            for i in range(0, 1):
                Kk = self.gain(alphak, Pk, Pkk)
                betak = self.bias(alphak, Pk, alphakk, Pkk)
                alphak = self.variationalBackwardMean(k-1, alphak, Pk, alphakk, Pkk,betak,Kk)
                Pk = self.variationalBackwardRiccati(alphak, Pk, Pkk, Kk)
            alphakk=alphak
            Pkk = Pk
            self.listK[k - 1,:] = Kk
            self.listbeta[k - 1,:] = betak
            self.listalpha[k - 1,:] = alphak
            self.listV[k - 1,:] = Pk
        super().synthetizeController()
        return self.listK, self.listV

    #@overrides
    def optimalPolicy(self,state,time):
        t=int(time/self.dt)
        K = self.listK[t]
        state=state.reshape(-1,1)
        alpha=self.listalpha[t]
        beta=self.listbeta[t]
        P = self.listV[t]
        mean_u=self.meanFeedback(state, alpha, beta,K)
        if self.randomControl:
            u=np.random.multivariate_normal(mean_u, self.cov(P), 1).reshape(self.dimControl,1)
        else:
            u=mean_u
        return u

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







