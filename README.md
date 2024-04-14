# Variational Dynamic Programming for Stochastic Optimal Control

## Object

This is the companion code for the paper \[[1] (To appear) \]. Please cite this paper if you use this code.  

## Installation
The code is available in python using the standard library. 

## Model
The stochastic dynamic is described by a Langevin SDE with a Brownian motion $W_t$ on the acceleration $a_t$:
$$dv_t=a_tdt=f(x_t,v_t,u_t)dt+L\sqrt{C}dW_t; \quad dx_t=v_tdt,$$
where 
- $v_t \in \mathcal{M}(d \times 1)$
- $u_t \in \mathcal{M}(p \times 1)$ 
- $W_t \in \mathcal{M}(m \times 1)$
- $L \in \mathcal{M}(d \times m)$ 
- $C \in \mathcal{M}(m \times m)$.
which is discretized using a semin-implicit scheme as follows:
$$v_{k+1}=v_k+f(x_t,v_t,u_t)\delta t+Lw_t; \quad x_{k+1}=x_k+v_{k+1}\delta t; \quad w_t \sim \mathcal{N}(0,C)$$

## python files
The source of the KL controller for the inverted pendulum problem.

[0]: https://arxiv.org/abs/ (To appear)

\[1\]: ["Variational Dynamic Programming for Stochastic Optimal Control.  Marc Lambert, Francis Bach, Silvère Bonnabel.Submitted to IEEE CDC 2024 "][4] 
