
###################################################################################
# THE VARIATIONAL CONTROL LIBRARY                                                 #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
###################################################################################
# Abstract controller                                                             #
###################################################################################

from collections.abc import Callable
from abc import ABCMeta, abstractmethod
import numpy as np

class DynamicCost:
    # The dual class emulating the cost function along a trajectory
    def __init__(self, transitionCost: Callable, finalCost: Callable,finalTime):
        self.transitionCost=transitionCost
        self.finalTime=finalTime
        self.finalCost=finalCost
        self.lcostToGo=[]
        self.loss = []
        self.loss.append(np.array([0,0]))
        self.dec=5 #time precision

    def computeLoss(self,currtraj, currControls):
        losst=0
        N=np.shape(currtraj)[0]
        for i in range(1,N):
            time = currtraj[i, 0]
            state = currtraj[i, 1:]
            if abs(time -self.finalTime)<1e-6:
                losst = losst + self.finalCost(state)
                self.loss.append(np.array([time,losst]))
            else:
                u = currControls[i, 1:]
                losst = losst + self.transitionCost(state, u, time)
                self.loss.append(np.array([time,losst]))
        return self.loss

    # Compute the cost to go along the whole path
    def computeCostToGo(self,currtraj, currControls):
        costt = 0
        N = np.shape(currtraj)[0]
        for i in range(N-1, 0, -1):
            time = currtraj[i, 0]
            state = currtraj[i, 1:]
            if abs(time -self.finalTime)<1e-6:
                costt = costt + self.finalCost(state)
                self.lcostToGo.append(np.array([time,costt]))
            else:
                u = currControls[i, 1:]
                costt = costt + self.transitionCost(state, u, time)
                self.lcostToGo.append(np.array([time,costt]))
        self.lcostToGo.reverse()
        return self.lcostToGo

class StochasticController:
    # a generic controller
    def __init__(self, system, dimControl, transitionCost: Callable, finalCost: Callable, finalTime):
        self.cost=DynamicCost(transitionCost, finalCost, finalTime)
        self.system=system
        self.dimControl=dimControl
        self.dt=system.dt

    @abstractmethod
    def synthetizeController(self):
        self.system.setPolicy(self.optimalPolicy)
        return

    @abstractmethod
    def optimalPolicy(self,state,time):
        return np.zeros([self.dimControl, 1])

    # Compute the value function from a state
    def costTogo(self, t, statet):
        self.system.reinitializeAt(t, statet)
        currtraj, currControls = self.system.propagate(self.cost.finalTime, addNoise=True)
        costToGo=self.cost.computeCostToGo(currtraj, currControls)
        return costToGo[0,0]

    # Compute the value along the whole path
    def costTogoHisto(self):
        self.system.reinitialize()
        currtraj, currControls = self.system.propagate(self.cost.finalTime, addNoise=True)
        return np.array(self.cost.computeCostToGo(currtraj, currControls))

    # Compute the value along the whole path
    def lossHisto(self):
        self.system.reinitialize()
        currtraj, currControls = self.system.propagate(self.cost.finalTime, addNoise=True)
        return np.array(self.cost.computeLoss(currtraj, currControls))








