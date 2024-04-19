###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# Abstrcact dynamic                                                               #
###################################################################################

import numpy as np
from collections.abc import Callable
from abc import ABCMeta, abstractmethod
import sympy as sym
from sympy import lambdify

class ContinuousStochasticDynamicalSystem:
    # Supposed in (under-damped) Langevin form x=[x,p] to use semi-implicit scheme:
    # p(t + dt) = p(t) + f(x,p, u)dt + L.N(0, dt.C)
    # x(t + dt) = x(t) + p(t+dt)dt
    # dim(x)=d; dim(u)=dim(noise)=m
    # the control u is set to 0 by default (passive dynamic)
    # and can be reset by the "setPolicy" method with a state-feedback callback
    def __init__(self, x0, p0, L, C,policy=[]):
        #self.state0 = np.concatenate((x0, p0), axis=0)#np.array([x0, p0])
        self.x0 = x0
        self.p0 = p0
        self.x = x0
        self.p = p0
        self.d=L.shape[0]
        self.m=L.shape[1]
        self.traj = []
        self.traj.append(np.concatenate((np.array([0]), x0,p0), axis=0))
        self.controls = []
        self.time = 0
        if policy==[]:
            self.policy=self.passivePolicy
        else:
            self.policy = policy
        self.state0=np.concatenate((x0, p0), axis=0)
        u0=self.policy(self.state0,0)
        self.dimu=u0.shape[0]
        self.L=L
        self.C=C

    @abstractmethod
    def controlledDynamic(self,x,p,u):
        return

    def passivePolicy(self,state,t):
        # by default p=d
        d=int(state.shape[0]/2)
        ut=np.zeros([d, 1])
        return ut

    def passiveDynamic(self,X):
        u=np.zeros([self.dimu,1])
        x_ = X[0:self.d]
        p_ = X[self.d:]
        return self.controlledDynamic(x_,p_,u)

    def setPolicy(self,policy: Callable):
        self.policy=policy
        u0=self.policy(self.state0,0)
        self.dimu=u0.shape[0]

    # reinitialize at default initial value
    def reinitialize(self):
        self.x = self.x0
        self.p=self.p0
        self.time = 0
        self.traj = []
        self.traj.append(np.concatenate((np.array([0]), self.x0,self.p0), axis=0))
        self.controls = []

    # reinitialize at arbitrary state and time (usefull for cost-to-go)
    def reinitializeAt(self,t,xt,pt):
        self.x = xt
        self.p = pt
        self.time = t
        self.traj = []
        self.traj.append(np.concatenate((np.array([0]), xt,pt), axis=0))
        self.controls = []

    @staticmethod
    def integrateSDE(x,p,f,u,L,C,dt,addNoise=True,method='Euler_SI'):
        m=C.shape[0]
        if method=='Euler':
            xn = x + p * dt
            a = f(x, p, u)
            pn = p + a * dt
            if addNoise:
                w = np.random.multivariate_normal(np.zeros([m, ]), dt * C, 1).reshape(-1,)
                pn = p + L.dot(w)
        elif method=='Euler_SI':
            a = f(x, p, u)
            pn = p + a * dt
            if addNoise:
                w = np.random.multivariate_normal(np.zeros([m, ]), dt * C, 1).reshape(-1,)
                pn = pn + L.dot(w)
            xn = x + pn * dt
        else:
            print("unknown method of integration: choose Euler or Euler_SI")
            return
        return xn,pn

    # propagation from t to t+T with semi-implicit integration
    def propagateCore(self, dt, T, addNoise=True):
        t=0
        while t < T:
            state=np.concatenate((self.x,self.p), axis=0)
            ut=self.policy(state,self.time)
            ut=ut.reshape(-1,)
            self.x,self.p=StochasticDynamicalSystem.integrateSDE(self.x,self.p,self.controlledDynamic,ut,self.L,self.C,dt,addNoise=addNoise)
            self.time = self.time + dt
            t=t+dt
            # python add extra decimals when adding float: to avoid it we round the results
            nbdec=5
            #self.x = np.round(self.x, nbdec)
            #self.p = np.round(self.p, nbdec)
            self.time = np.round(self.time, nbdec)
            t = np.round(t, nbdec)
            stateTime=np.concatenate((np.array([self.time]), state), axis=0)
            self.traj.append(stateTime)
            control=np.concatenate((np.array([self.time]), ut), axis=0)
            self.controls.append(control)
            #print("traj at time {}, pos={} m, vel={} m/s, control={} m/s2".format(self.time, self.x, self.p, ut))
        return np.array(self.traj), np.array(self.controls)

    # trajectory in phase space
    def trajArr(self):
        return np.array(self.traj)

    def controlArr(self):
        return np.array(self.controls)


class StochasticDynamicalSystem(ContinuousStochasticDynamicalSystem):
    # Discrete-Time system deduced from Continuous system
    # Xk+1=F(Xk)+BUk where F is called the discrete Transition
    def __init__(self, x0, p0, L, C, dt, policy=[]):
        super().__init__(x0, p0, L, C, policy)
        self.dt=dt

    def propagate(self, T, addNoise=True):
        return self.propagateCore(self.dt, T, addNoise)

    # To compute the symbolics Jacobian and Hessian of the transition
    def initializeAuto(self):
        Nx = 2 * self.d
        X = sym.symbols("X:{:}".format(Nx))
        self.F = lambdify([X], self.discreteTransitionAuto(X), "numpy")
        jac = self.discreteTransitionAuto(X).jacobian(X)
        self.Fx = lambdify([X], jac, "numpy")
        self.Fxx = []
        for i in range(Nx):
            self.Fxx.append(lambdify([X], jac.row(i).jacobian(X), "numpy"))

    def discreteTransition(self, X):
        return

    # discrete-transition F st Xk+1=F(Xk) (without control)
    def discreteTransitionAuto(self, X):
        x_=X[0:self.d]
        x_ = sym.Matrix([xi for xi in x_])
        p_=X[self.d:]
        p_ = sym.Matrix([pi for pi in p_])
        u_ = np.zeros([self.dimu,1])
        a=self.controlledDynamic(x_,p_,u_)
        xn = x_ + p_ * self.dt
        pn = p_ + a * self.dt
        return xn.col_join(pn)

    def jacobianTransition(self, X):
        return

    def jacobianTransitionAuto(self, X):
        return self.Fx(X)

    def hessianTransition(self, X):
        return

    def hessianTransitionAuto(self, X):
        Hess = np.zeros([2 * self.d, 2 * self.d, 2 * self.d])
        for i in range(0, 2 * self.d):
            Hess[i, :, :] = self.Fxx[i](X)
        return Hess


class LinearSystem(ContinuousStochasticDynamicalSystem):

    #@overrides
    def controlledDynamic(self,x,p,u):
        state = np.concatenate((x,p), axis=0)
        return self.F.dot(state)+self.Bp.dot(u)

    def passivePolicy(self, state, t):
        ut = np.zeros([self.Bp.shape[0], 1])
        return ut

    def __init__(self,x0,p0,L,C,F,Bp):
        self.F=F #Ap:(x,p)->dot p
        self.Bp=Bp #Bp:(u)->dot p
        p=Bp.shape[1]
        passivePolicy = lambda X, t: np.zeros([p, 1])
        super().__init__(x0,p0, L, C,passivePolicy)







