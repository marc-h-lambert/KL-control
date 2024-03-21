
###################################################################################
# LQR and variational LQR control of a simple Pendulum                            #
# Code supported by Marc Lambert                                                  #
###################################################################################
# Reproduce the results of the paper                                              #
# "Variational Dynamic Programming for Stochastic Optimal Control"                          #
# Authors: Marc Lambert, Francis Bach, Silvère Bonnabel                           #
###################################################################################
# TO DO: split in classes and use automatic-differentiation to test others systems
###################################################################################

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

plt.rcParams['pdf.fonttype'] = 42

###########################################################################################################
# Pendulum dynamic
###########################################################################################################

########## PENDULE Parameters ############
m = 1      # mass of pendulum
l = 1   # length of pendulum
g = 9.8     # gravitational constant
n=-1 # n=1 for pendulum or n=-1 for inverted pendulum
dt=0.01 # step size (takes 0.01 for Euler-Marayama)
N=1000 # nb Iterations
xi=1. # damping
deg=math.pi/180

########## INITIAL VALUE & TARGET ############
frac=6
x0=np.array([np.pi/frac,0]).reshape(2,1)
xT=np.array([0,0]).reshape(2,1)

B=np.array([[0], [1/(m*l*l)]])*dt # control matrix
Bw=np.array([[0], [1]]) # beyond the noise

# TRANSITION FUNCTION
def f(X):
    x=X[0]
    y=X[1]
    return np.array([x+dt*y,y- dt * y * xi/m-g/l*n*dt*math.sin(x)])

# TRANSITION JACOBIAN
def Jf(X):
    x=X[0]
    y=X[1]
    return np.array([[1,dt],[-g/l*n*dt*math.cos(x),1-dt*xi/m]])

# DYNAMIC FUNCTION
def f_d(X):
    x=X[0]
    y=X[1]
    return np.array([y,-y * xi/m-g/l*n*math.sin(x)])

###########################################################################################################
# Linearized LQR control
###########################################################################################################

def gainLQR(A,P,R,B):
    S=R+B.T.dot(P).dot(B)
    return LA.inv(S).dot(B.T).dot(P).dot(A)

def iterateLQR(A,P,R,B,Q):
    S=R+B.T.dot(P).dot(B)
    return Q-A.T.dot(P).dot(B).dot(LA.inv(S)).dot(B.T).dot(P).dot(A)+A.T.dot(P).dot(A)

def BackwardPassLQR(A,B,Q,R,PT,xT,N):
    listK = np.zeros([N, 2])
    listP = np.zeros([N + 1, 2, 2])
    listP[N] = PT
    P=PT
    xg=xT
    for i in range(N, 0, -1):
        # print("--- iteration N°", i)
        K = gainLQR(A, P, R, B)
        P = iterateLQR(A, P, R, B, Q)
        listK[i - 1] = K
        listP[i - 1] = P
    return listK, listP



###########################################################################################################
# Variational control given by the variational backward Riccati equation
###########################################################################################################


########## Functions to compute expectations with cubbature points ############
def H(X, K, P,n,delta):
  x = X[0]
  y = X[1]
  s = P[1, 0] * (x + delta * y) + P[1, 1] * (y + y * delta * xi/m - n * delta * g/l * math.sin(x) - delta*K.dot(X))
  res = np.zeros((2, 2))
  res[0, 0] = s * n * delta * g/l * math.sin(x)
  return res

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

def JPJ(x,Jf,P,*args):
    return Jf(x,*args).T.dot(P).dot(Jf(x,*args))

########## variational backward Riccati equation ############

def gainKL(xk,Pk,Pkk,R,B,Jfd,eps):
    S=R+B.T.dot(Pkk).dot(B)
    # compute the expectation with cubature points
    EA = fmeanCKF(Jfd, xk, LA.cholesky(LA.inv(Pk) * eps))
    return LA.inv(S).dot(B.T).dot(Pkk).dot(EA)

def iterateKL(Pk,Pkk,xk,R,B,Q,Jfd,eps):
    covV = LA.inv(Pk) * eps
    sqrtV=LA.cholesky(covV)
    S=R+B.T.dot(Pkk).dot(B)
    A =fmeanCKF(Jfd, xk, sqrtV)
    M1 = fmeanCKF(JPJ,xk, sqrtV,Jfd,Pkk)
    K = gainKL(xk,Pk,Pkk, R, B, Jfd, eps)
    M2 = fmeanCKF(H,xk, sqrtV,K,Pkk,n,dt)
    return Q - A.T.dot(Pkk).dot(B).dot(LA.inv(S)).dot(B.T).dot(Pkk).dot(A) + M1 +M2

def BackwardPassKL(Jacf,B,Q,R,PT,xT,N,eps=1e-6,Ninner=10):
    listK = np.zeros([N, 2])
    listP = np.zeros([N + 1, 2, 2])
    listP[N] = PT
    Pkk=PT
    for i in range(N, 0, -1):
        # print("--- iteration N°", i)
        ## Inner loops to solve the implicit scheme (we start with LQR guess)
        Pk = iterateLQR(Jacf(xT), Pkk, R, B, Q)
        for k in range(0, Ninner):
            Pk = iterateKL(Pk, Pkk, xT, R, B, Q, Jacf, eps)
        K = gainKL(xT, Pk, Pkk, R, B, Jacf, eps)
        Pkk=Pk
        listK[i - 1] = K
        listP[i - 1] = Pk
    return listK, listP

###########################################################################################################
# SIMULATION
# we simulate here a control-affine dynamic forward using a control law defined by list of gains listK and list
# of values listP.
###########################################################################################################
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

# x0, f, eta --> iniial state and dynamic and transition noise covariance
# B,R, listK, listP --> parameters of the optimal control law (at each time-index)
# epsKL --> the regularization parameter (only used for maxEnt policy)
# if not sampleControl average control else : Random control (only used for maxEnt policy)
def simulate(dt, x0, f, B, R, N, listK, listP, constrained=False,eta=0,epsKL=0,sampleControl=False,seed=1,KL=True):
    x = x0 # initial state
    traj = np.zeros([N, 2]) # to keep track on the traj
    controls = np.zeros([N]) # to keep track on the controls
    loss = np.zeros([N]) # to keep track on the loss
    valuex = np.zeros([N]) # to keep track on the cost-to-go
    for i in range(0,N):
        traj[i,:] = x.reshape(2,)
        K=listK[i].reshape(1,2)
        P = listP[i]
        invP=LA.inv(P)
        S = R + B.T.dot(P).dot(B)
        sigma_u=np.sqrt(epsKL*LA.inv(S))
        mean_u=-K.dot(x)
        if sampleControl:
            u=np.random.normal(mean_u,sigma_u)
        else:
            u = mean_u
        if constrained:
            u=np.clip(u,-2,2)
            #u=np.tanh(u)
        controls[i] = u

        if not KL:
            valuex[i] = x.T.dot(P).dot(x)
            if i > 0:
                loss[i] = loss[i - 1] + x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
            else:
                loss[i] = x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
        else:
            (sign, logdetinvP) = np.linalg.slogdet(epsKL*invP)
            logq=-0.5*x.T.dot(P).dot(x)/epsKL-0.5*logdetinvP-math.log(2*math.pi)
            #q=multivariate_normal.pdf(x=x.reshape(2,), mean=np.zeros([2,]), cov=invP*epsKL)
            valuex[i]=-epsKL*logq
            if i > 0:
                 k= x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
                 loss[i] = loss[i - 1]*math.exp(-k/epsKL)
            else:
                k= x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
                loss[i] = math.exp(-k / epsKL)
        if not is_pos_def(P):
            print("MATRIX NOT SDP !!")
        # We can not use a Runge-Kutta scheme for stochastic dynamic
        # We use the Euler-Maruyama variant with semi-implicit scheme  to better conserve energy
        xtt = np.zeros([2,1 ])
        if eta>0:
            w=np.random.normal(0,np.sqrt(eta*dt))
            out = f(x) + B.dot(u).reshape(2,1) + Bw.dot(w)
            xtt[1]=out[1]
            xtt[0]=x[0]+dt*xtt[1]
        else:
            out = f(x) + B.dot(u).reshape(2, 1)
            xtt[1] = out[1]
            xtt[0] = x[0] + dt * xtt[1]
        x=xtt

    return traj, controls, loss, valuex


###########################################################################################################
# PLOT AVERAGE CONTROL
###########################################################################################################
def plotControl(listK,traj,controls,loss,ax,eps,col,KL=True,ls='-'):
    T=np.linspace(0, dt * N, N)
    if KL:
        ax[0].plot(T, traj[:,0]*180/np.pi, label=r'$\sqrt{\varepsilon}=$'+'{:,.2f}'.format(math.sqrt(eps)),color=col,linestyle=ls,linewidth='3')
    else:
        ax[0].plot(T, traj[:, 0] * 180 / np.pi,color=col,linestyle=ls,linewidth='3')
    ax[1].plot(T, controls,color=col,linestyle=ls,linewidth='3')
    ax[2].plot(T, np.array(listK)[:,0],color=col,linestyle=ls,linewidth='3')
    ax[2].plot(T, np.array(listK)[:,1],color=col,linestyle=ls,linewidth='3')
    ax[3].plot(T, loss, color=col,linestyle=ls,linewidth='3')
    if n==1:
        ax[0].set_ylabel(r"pendulum angle $\theta^\circ$",weight='bold')
    elif n==-1:
        ax[0].set_ylabel(r"pendulum angle $\theta^\circ$",weight='bold')
    ax[1].set_ylabel("controls",weight='bold')
    ax[2].set_ylabel("Gains",weight='bold')
    ax[3].set_ylabel("Cumulative loss",weight='bold')

###########################################################################################################
# PLOT RANDOM CONTROL
###########################################################################################################
def plotStatsTraj(trajMC, controlsMC, valueMC,listP,ax):
    T=np.linspace(0, dt * N, N)
    # ----------- Plot Traj ------------
    meanTraj = np.mean(trajMC, axis=2)[:,0]*180/np.pi
    stdTraj = np.sqrt(np.var(trajMC, axis=2))[:,0]*180/np.pi
    ax[0].plot(T, meanTraj, linewidth='3',label='mean',color='r')
    ax[0].fill(np.append(T, T[::-1]), np.append(meanTraj+stdTraj, (meanTraj-stdTraj)[::-1]), 'darkgray')
    if n==1:
        ax[0].set_ylabel(r"pendulum angle $\theta^\circ$",weight='bold')
    elif n==-1:
        ax[0].set_ylabel(r"pendulum angle $\theta^\circ$",weight='bold')
    # ----------- Plot Control ------------
    meanControl = np.mean(controlsMC, axis=1)
    stdControl = np.sqrt(np.var(controlsMC, axis=1))
    listdP=[]
    for i in range(1, N):
        listdP.append(listP[i] - listP[i - 1])
    ax[1].plot(T, meanControl, linewidth='3', label='mean', color='r')
    ax[1].fill(np.append(T, T[::-1]), np.append(meanControl + stdControl, (meanControl - stdControl)[::-1]), 'darkgray')
    ax[1].set_ylabel(r"control ($\circ/s^2$)", weight='bold')

    # ----------- Plot Value ------------
    meanValue = np.mean(valueMC,axis=1)
    stdValue = np.sqrt(np.var(valueMC, axis=1))
    ax[2].plot(T, meanValue, linewidth='3',label='mean',color='r')
    ax[2].fill(np.append(T, T[::-1]), np.append(meanValue+stdValue, (meanValue-stdValue)[::-1]), 'darkgray')
    ax[2].set_ylabel("cost-to-go", weight='bold')
    ax[2].set_xlabel("time (s)",weight='bold')


###########################################################################################################
# MAIN PROGRAM
###########################################################################################################
if __name__ == "__main__":
    TEST=["Test1_LQR","Test1_KL","Test2_LQR","Test2_KL"]
    num=0
    seed=10
    eps_kl=np.array([1*deg, 4*deg, 6*deg])
    eps_kl = eps_kl * eps_kl
    Nmc=30 # Nb MC trials

    ########## LOSS ############
    # INIT LOSS
    r = 0.01
    q = 0.01
    # we don't see the influence on eps
    Q = np.identity(2) * q * dt
    R = np.identity(1) * r * dt

    ###########################################################################################################
    # TEST AVERAGE CONTROL
    ###########################################################################################################
    if "Test1_LQR" in TEST:
        print("---------- Test1 LQR --------")
        eta = 1 * deg * 1 * deg
        np.random.seed(seed)
        # INIT Final Cost
        p = 0.01
        PT = np.identity(2) * p
        # Test LQR
        A = Jf(xT.reshape(2, ))
        num = num + 1
        plt.figure(num)
        fig, ax = plt.subplots(4, 1, figsize=(4, 6))
        for i in range(0,1):
            col='b'
            listK, listP = BackwardPassLQR(A,B,Q,R,PT,xT,N)
            print("K-LQR-discret=", listK[0])
            print("P-LQR-discret=", listP[0])
            traj,controls,loss,value=simulate(dt, x0, f, B, R, N, listK, listP,eta=eta,seed=seed,KL=False)
            plotControl(listK,traj,controls,loss,ax,0,col,KL=False)
        if n == 1:
            FileName = "Test1_LQR_Pendulum"
            ax[0].set_title("LQR control \n (Pendulum)",weight='bold')
        elif n == -1:
            FileName = "Test1_LQR"
            ax[0].set_title("LQR control",weight='bold')
        plt.tight_layout()
        plt.savefig(FileName + ".pdf", format="pdf")


    if "Test1_KL" in TEST:
        eta = 1 * deg * 1 * deg
        np.random.seed(seed)
        # INIT Final Cost
        p = 0.01
        PT = np.identity(2) * p
        # Test KL
        num = num + 1
        plt.figure(num)
        fig, ax = plt.subplots(4, 1, figsize=(4, 6))
        colors = np.array(["b", "g", "r"])
        styles=np.array(["-", "--", "-."])
        for i in range(0,eps_kl.shape[0]):
            epsKL = eps_kl[i]
            print(r"---------- Test1 KL - sqrt-eps={0:.2f} --------".format(math.sqrt(epsKL)))
            col=colors[i]
            ls=styles[i]
            listK, listP = BackwardPassKL(Jf,B,Q,R,PT,xT,N,eps=epsKL,Ninner=10)
            print("K-KL-discret=", listK[0])
            print("P-KL-discret=", listP[0])
            traj,controls,loss,value=simulate(dt, x0, f, B, R, N, listK, listP,eta=eta,epsKL=epsKL,sampleControl=False,seed=seed,KL=False)
            plotControl(listK,traj,controls,loss,ax,epsKL,col,KL=True,ls=ls)
        ax[0].legend()
        if n==1:
            FileName="Test1_KL_Pendulum"
            ax[0].set_title("mean Variational control \n (Pendulum)",weight='bold')
        elif n==-1:
            FileName = "Test1_KL"
            ax[0].set_title("mean Variational control",weight='bold')
        plt.tight_layout()
        plt.savefig(FileName + ".pdf", format="pdf")

    ###########################################################################################################
    # TEST RANDOM CONTROL
    ###########################################################################################################
    if "Test2_LQR" in TEST:
        eta = 6 * deg * 6 * deg
        np.random.seed(seed)
        # INIT Final Cost
        p = dt*1000
        PT = np.identity(2) * p
        # Test KL
        num = num + 1
        plt.figure(num)
        A = Jf(xT.reshape(2, ))
        fig, ax = plt.subplots(3, 1, figsize=(4, 6))
        trajMC = np.zeros([N, 2,Nmc])
        lossMC = np.zeros([N, Nmc])
        controlsMC = np.zeros([N,Nmc])
        valueMC = np.zeros([N,Nmc])
        for k in range(0,Nmc):
            print("---------- Test2 LQR - run({}) --------".format(k))
            listK, listP = BackwardPassLQR(A,B,Q,R,PT,xT,N)
            print("K-LQR-discret=", listK[0])
            print("P-LQR-discret=", listP[0])
            traj,controls,loss,value=simulate(dt, x0, f, B, R, N, listK, listP,eta=eta,seed=seed,KL=False)
            trajMC[:,:,k]=traj
            lossMC[:,k]=loss
            controlsMC[:,k]=controls
            valueMC[:,k]=value
        plotStatsTraj(trajMC,controlsMC, valueMC,listP,ax)
        if n==1:
            FileName="Test2_LQR_Pendulum"
            ax[0].set_title("LQR control \n (Pendulum)",weight='bold')
        elif n==-1:
            FileName = "Test2_LQR"
            ax[0].set_title("LQR control",weight='bold')
        plt.tight_layout()
        plt.savefig(FileName + ".pdf", format="pdf")

    if "Test2_KL" in TEST:
        eta = 6 * deg * 6 * deg
        np.random.seed(seed)
        # INIT Final Cost
        p = dt*1000
        PT = np.identity(2) * p
        # Test KL
        num = num + 1
        plt.figure(num)
        fig, ax = plt.subplots(3, 1, figsize=(4, 6))
        epsKL = eps_kl[-1]
        trajMC = np.zeros([N, 2,Nmc])
        lossMC = np.zeros([N, Nmc])
        controlsMC = np.zeros([N,Nmc])
        valueMC = np.zeros([N,Nmc])
        for k in range(0,Nmc):
            print("---------- Test2 KL - run({}) --------".format(k))
            listK, listP = BackwardPassKL(Jf,B,Q,R,PT,xT,N,eps=epsKL,Ninner=10)
            print("K-KL-discret=", listK[0])
            print("P-KL-discret=", listP[0])
            traj,controls,loss,value=simulate(dt, x0, f, B, R, N, listK, listP,eta=eta,epsKL=epsKL,sampleControl=True,seed=seed,KL=True)
            trajMC[:,:,k]=traj
            lossMC[:,k]=loss
            controlsMC[:,k]=controls
            valueMC[:,k]=value
        plotStatsTraj(trajMC,controlsMC, valueMC,listP,ax)
        ax[0].legend()
        if n==1:
            FileName="Test2_KL_Pendulum"
            ax[0].set_title("random Variational control \n (Pendulum)",weight='bold')
        elif n==-1:
            FileName = "Test2_KL"
            ax[0].set_title("random Variational control",weight='bold')
        plt.tight_layout()
        plt.savefig(FileName + ".pdf", format="pdf")

plt.show()






