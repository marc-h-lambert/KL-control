# Variational Dynamic Programming for Stochastic Optimal Control

## Object

This is the companion code for the paper \[[1] (To appear) \]. Please cite this paper if you use this code.  

## Installation
The code is available in python using the standard library. 

## Model
The stochastic dynamic is described by a Langevin SDE with a Brownian motion $W_t$ on the acceleration $a_t$:
$$dv_t=a_tdt=f(x_t,v_t)dt +Bu_tdt+L\sqrt{C}dW_t; \quad dx_t=v_tdt,$$
where 
- $x_t \in \mathcal{M}(d \times 1)$, $v_t \in \mathcal{M}(d \times 1)$ and $a_t \in \mathcal{M}(d \times 1)$ are the position, velocity and acceleration vectors
- $u_t \in \mathcal{M}(p \times 1)$ is the control vector with $p \leq d$
- $W_t \in \mathcal{M}(m \times 1)$ is the noise vector with $m \leq d$
- $B \in \mathcal{M}(d \times p)$ is the control matrix (acceleration-channel where enter the control)
- $C \in \mathcal{M}(m \times m)$ is the covariance of the noise
- $L \in \mathcal{M}(d \times m)$ is the noise matrix (acceleration-channel where enter the noise).
  
The dynamic is integrated with a step $dt$ with a semi-implicit scheme:
$$v_{k+1}=v_k+f(x_t,v_t)dt+Bu_tdt+Lw_t; \quad x_{k+1}=x_k+v_{k+1}dt; \quad w_t \sim \mathcal{N}(0,C)$$

## python files
The source of the KL controller for the inverted pendulum problem.

[0]: https://arxiv.org/abs/ (To appear)

\[1\]: ["Variational Dynamic Programming for Stochastic Optimal Control.  Marc Lambert, Francis Bach, Silvère Bonnabel.Submitted to IEEE CDC 2024 "][4] 
