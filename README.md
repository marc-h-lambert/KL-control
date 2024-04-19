# Variational Dynamic Programming for Stochastic Optimal Control

## Object

This is the companion code for the paper \[[1] (To appear) \]. Please cite this paper if you use this code.  

## Running code
You can run the file \[[VariationalDP][2]\] which contains all the code in one file 
to reproduce the results of the paper. The code has been rewritten in a library-class \[[ControlLibrary][3]\]  for easier support: the classes are tested 
in \[[TestPendulumLQR][4]\]  for the LQR regulator and in \[[TestPendulumKLcontrol][5]\] for the Variational controller.

## Installation
The standalone file \[[VariationalDP][2]\]  is available in python using the standard library. 
The control library embeds dependencies on Sympy (https://docs.sympy.org/latest/install.html) 
for the differentiation of the gradient and the Hessian. 

## Model of the stochastic dynamic
### Affine control:
The system is described by a Langevin SDE with a Brownian motion $W_t$ supposed to disrupt acceleration:
$$dv_t=a_tdt=f(x_t,v_t)dt +B_pu_tdt+L\sqrt{C}dW_t; \quad dx_t=v_tdt,$$
where 
- $x_t,v_t,a_t \in \mathbb{R}^d$ are the position, velocity and acceleration vectors (we use in the code the momentum notation p_t:=v_t)
- $f:\mathbb{R}^d\times \mathbb{R}^d \rightarrow \mathbb{R}^d$ is the (known) dynamic of the system
- $u_t \in \mathbb{R}^p$ is the control vector with $p \leq d$
- $W_t \in \mathbb{R}^m$ is the noise vector with $m \leq d$
- $B_p \in \mathcal{M}(d \times p)$ is the control matrix (acceleration-channel where enter the control)
- $C \in \mathcal{M}(m \times m)$ is the covariance of the noise
- $L \in \mathcal{M}(d \times m)$ is the noise matrix (acceleration-channel where enter the noise).
  
The dynamic is integrated with a step $\mathrm{dt}$ with a semi-implicit Euler-Maruyama scheme:
$$v_{k+1}=v_k+f(x_k,v_k)\mathrm{dt}+Bu_k\mathrm{dt}+Lw_k; \quad x_{k+1}=x_k+v_{k+1}\mathrm{dt}; \quad w_k \sim \mathcal{N}(0,C\mathrm{dt})$$

### Linear case:
In the linear case we have $f(x_t,v_t)=F \pmatrix{x_t \\\ v_t}$ where $F \in \mathcal{M}(d \times 2d)$ and the corresponding discrete-time state-space representation is:
$$\pmatrix{x_{k+1} \\\ v_{k+1}}= \pmatrix{x_{k} \\\ v_{k}} + \pmatrix{ (0, I_d) \\\ F }  \pmatrix{x_{k} \\\ v_k}\mathrm{dt} + \pmatrix{(0) \\\ B_p }  u_k \mathrm{dt} +\pmatrix{(0) \\\ Lw_k},$$
which takes the form $X_{k+1} =\mathcal{A}X_k+ \mathcal{B}u_k+\nu_k$ where $X_k=\pmatrix{x_{k} \\\ v_k}$, $\mathcal{A}=I_{2d} + \pmatrix{ (0, I_d) \\\ F }\mathrm{dt}$,  $\mathcal{B}=\pmatrix{(0) \\\ B_p} \mathrm{dt}$ and $\nu_k=\pmatrix{(0) \\\ Lw_k}$.

The model is specified with $F$ and $B_p$ in continuous-time but solved by LQR using $\mathcal{A}$ and $\mathcal{B}$ in discrete-time.

## Model of the loss
The controller is minimizing the finite-time continuous loss: $\int_0^T \ell(x_t,v_t,u_t,t)dt +V_T(x_T,v_T)$ which writes in the discrete-time-state-space representation with a step $\mathrm{dt}$ and with $N:=T/\mathrm{dt}$:
$$\sum_{k=0}^{N-1} \ell(X_k,u_k,k\mathrm{dt}) \mathrm{dt} + V_N(X_N)$$

### LQR case:
In the LQR case and for our variational controler the loss is:
$$\sum_{k=0}^{N-1} (X_k^T Q X_k+u_k^TR   u_k)\mathrm{dt} + X_N^TV_NX_N,$$
where
- $Q \in \mathcal{M}(2d \times 2d)$ is the matrix of the quadratic form associated with the transition state-cost 
- $R \in \mathcal{M}(m \times m)$ is the matrix of the quadratic form associated with the transition control cost (cost in energy) 
- $V_N \in \mathcal{M}(2d \times 2d)$ is the matrix of the quadratic form associated with the instantaneous final cost (or final Value).
  
## python files
The source of the KL controller for the inverted pendulum problem.

[0]: https://arxiv.org/abs/ (To appear)

\[1\]: ["Variational Dynamic Programming for Stochastic Optimal Control.  Marc Lambert, Francis Bach, Silvère Bonnabel.Submitted to IEEE CDC 2024 "][4] 

[2]: ./VariationalDP.py
[3]: ./ControlLibrary
[4]: ./TestPendulumLQR.py
[5]: ./TestPendulumKLcontrol.py
