# Variational Dynamic Programming for Stochastic Optimal Control

## Object
The control library is thought to simulate a general stochastic dynamic and 
to design a stochastic controller. The abstract classes are in \[[Core][0]\], they provide methods to compute automatically the gradient and the Hessian of the transition function with symbolic calculation. 
For the Pendulum dynamic, this automatic computation is not required (and is slower) since the gradient and Hessian are also computed explicitly in the Pendulum class. 

The abstract stochastic dynamic is specified in \[[StochasticDynamicsExample][1]\] including the Pendulum and the constant velocity model in 2D. The abstract controller is specified 
as a LQR controller  in \[[LQR][2]\] and as a variational controller in \[[KLcontrol][3]\].

[0]: ./Core
[1]: ./StochasticDynamicsExample.py
[2]: ./LQR.py
[3]: ./KLcontrol.py
