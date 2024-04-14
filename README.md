# Variational Dynamic Programming for Stochastic Optimal Control

## Object

This is the companion code for the paper \[[1] (To appear) \]. Please cite this paper if you use this code.  

## Installation
The code is available in python using the standard library. 

## Model of the stochastic dynamic
### Affine control:
The stochastic dynamic is described by a Langevin SDE with a Brownian motion $W_t$ supposed to disrupt acceleration:
$$dv_t=a_tdt=f(x_t,v_t)dt +Bu_tdt+L\sqrt{C}dW_t; \quad dx_t=v_tdt,$$
where 
- $x_t,v_t,a_t \in \mathbb{R}^d$ are the position, velocity and acceleration vectors
- $f:\mathbb{R}^d\times \mathbb{R}^d \rightarrow \mathbb{R}^d$ is the dynamic
- $u_t \in \mathbb{R}^p$ is the control vector with $p \leq d$
- $W_t \in \mathbb{R}^m$ is the noise vector with $m \leq d$
- $B \in \mathcal{M}(d \times p)$ is the control matrix (acceleration-channel where enter the control)
- $C \in \mathcal{M}(m \times m)$ is the covariance of the noise
- $L \in \mathcal{M}(d \times m)$ is the noise matrix (acceleration-channel where enter the noise).
  
The dynamic is integrated with a step $\mathrm{dt}$ with a semi-implicit Euler-Maruyama scheme:
$$p_{k+1}=p_k+f(x_k,p_k)\mathrm{dt}+Bu_k\mathrm{dt}+Lw_k; \quad x_{k+1}=x_k+p_{k+1}\mathrm{dt}; \quad w_k \sim \mathcal{N}(0,C\mathrm{dt})$$

### Linear case:
In the linear case we have $f(x_t,v_t)=F \pmatrix{x_t \\\ v_t}$ where $F \in \mathcal{M}(d \times 2d)$ and the corresponding discrete-time state-space representation is:
$$\pmatrix{x_{k+1} \\\ p_{k+1}}= \pmatrix{x_{k} \\\ p_{k}} + \pmatrix{ (0, I_d) \\\ F }  \pmatrix{x_{k} \\\ p_{k}}\mathrm{dt} + \pmatrix{(0) \\\ B }  u_k \mathrm{dt} +\pmatrix{(0) \\\ Lw_k},$$
which takes the form $X_{k+1} =\mathcal{A}X_k+ \mathcal{B}u_k+\nu_k$ where $X_k=\pmatrix{x_{k} \\\ p_{k}}$, $\mathcal{A}=I_{2d} + \pmatrix{ (0, I_d) \\\ F }\mathrm{dt}$,  $\mathcal{B}=\pmatrix{(0) \\\ B} \mathrm{dt}$ and $\nu_k=\pmatrix{(0) \\\ Lw_k}$.

## Model of the controller
The controller is minimizing the finite-time continuous loss: $\int_0^T \ell(x_t,v_t,u_t,t)dt +V_T(x_T,v_t)$ which writes in the discrete-time-state-space representation with a step $\mathrm{dt}$ and with $N:=T/\mathrm{dt}$:
$$\sum_{k=0}^{N-1} \ell(X_k,u_k,k\mathrm{dt}) \mathrm{dt} + V_N(X_N)$$

### LQR case:
In the LQR case the loss is:
$$\sum_{k=0}^{N-1} X_k^T(Q \mathrm{dt})X_k+u_k^T(R \mathrm{dt} ) u_k + X_N^TV_NX_N$$

## python files
The source of the KL controller for the inverted pendulum problem.

[0]: https://arxiv.org/abs/ (To appear)

\[1\]: ["Variational Dynamic Programming for Stochastic Optimal Control.  Marc Lambert, Francis Bach, Silvère Bonnabel.Submitted to IEEE CDC 2024 "][4] 
