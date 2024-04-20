
###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# Pendulum dynamic                                                                #
###################################################################################

import math
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from Core.StochasticDynamicalSystem import StochasticDynamicalSystem, LinearSystem

plt.rcParams['pdf.fonttype'] = 42

class Pendulum(StochasticDynamicalSystem):
    g = 9.81
    def systemDynamic(self,x,p):
        return -1*p[0] * self.ksi / self.m - Pendulum.g / self.l * self.n * math.sin(x[0])

    def passiveDynamicAuto(self,X):
        x = X[0:self.d]
        p = X[self.d:]
        return -1*p[0] * self.ksi / self.m - Pendulum.g / self.l * self.n * sym.sin(x[0])

    def controlledDynamic(self,x,p,u):
        B = np.array([[1 / (self.m * self.l * self.l)]])  # control matrix
        return self.systemDynamic(x,p)+B.dot(u)

    def policy(self,state,t):
        return 0

    #overrides(StochasticDynamicalSystem)
    def jacobianDynamic(self,X):
        Jacobianf= np.array([[-Pendulum.g/self.l*self.n*math.cos(X[0]),-self.ksi/self.m]])
        return Jacobianf

    def hessianDynamic(self, X):
        Hessianf = np.zeros((2, 2, 2))
        Hessianf[1, 0, 0] = self.n * math.sin(X[0]) * self.g / self.l
        return Hessianf

    # used only for validation
    def HmatrixValidation(self,X, K, P, dt):
        x = X[0]
        y = X[1]
        s = P[1, 0] * (x + dt * y) + P[1, 1] * (
                    y + y * dt * self.ksi / self.m - self.n * dt * self.g / self.l * math.sin(x) - dt * K.dot(X))
        res = np.zeros((2, 2))
        res[0, 0] = s * self.n * dt * self.g / self.l * math.sin(x)
        return res

    def __init__(self,m,l,ksi,theta0,dtheta0,eta,dt,invertedPendulum=True):
        self.m=m # pendulum mass
        self.l=l # pendulum length
        self.ksi=ksi # damping
        x0=np.array([theta0])# pendulum initial state
        p0 = np.array([dtheta0]) # pendulum initial state
        C=eta*np.identity(1)  # covariance noise
        L=np.identity(1) # entry matrix for noise
        if invertedPendulum:
            self.n=-1
        else:
            self.n=1
        super().__init__(x0,p0, L, C,dt)


class MRU(LinearSystem):
    def __init__(self,x0,p0,L,C,Bp):
        d = L.shape[0]
        m = L.shape[1]
        Ap=np.zeros([d,2*d])
        super().__init__(x0,p0, L, C,Ap,Bp)










