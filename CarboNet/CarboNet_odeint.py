"""
Implementation of CarboNet, the compartmental network 
proposed in [1] for tropospheric carbon control.
This code is part of the source code of [2].

References:
    [1] Zocco, F., Haddad, W.M. and Malvezzi, M., 2025. 
        CarboNet: A finite-time combustion-tolerant 
        compartmental network for tropospheric carbon 
        control. arXiv preprint arXiv:2508.16774.
    [2] Zocco, F., Corti, A. and Malvezzi, M., 2025. CiRL: 
        Open-source environments for reinforcement learning 
        in circular economy and net zero. arXiv preprint 
        arXiv:2505.21536. 
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#######################Simulator settings#####################
# Simulation time:
t_final = 3*60


# Model parameters:
# For eq. 1:
a_41 = 0.2
a_14 = 0.1
n_h = 10000
a_13 = 0.5/n_h
n_q = 5000
a_12 = 0.5/n_q
# For eq. 2:
a_22 = 0.3
a_42 = 0.5/n_q
# For eq. 3:
a_33 = 0.6
a_43 = 0.5/n_h
# For eq. 4:
# no further parameters


# Initial conditions:
x1_ini = 915.4
x2_ini = 210.0
x3_ini = 500.0
x4_ini = 1830.8
X_ini = np.array([x1_ini, x2_ini, x3_ini, x4_ini])

# Inputs: 
u = 20 # arbitrary choice for code testing

def CarboNet(X, t=0):
    
    return np.array([- a_41*X[0] + a_14*X[3] + n_h*a_13*X[2] + n_q*a_12*X[1] - u,
                     a_22*X[1] - n_q*a_12*X[1] - n_q*a_42*X[1],
                     a_33*X[2] - n_h*a_13*X[2] - n_h*a_43*X[2],
                     a_41*X[0] - a_14*X[3] + n_q*a_42*X[1] + n_h*a_43*X[2]])


# Numerical solution:
t = np.linspace(0, t_final, 1000)
X, infodict = integrate.odeint(CarboNet, X_ini, t,
mxstep = 1000, full_output = True)
x1, x2, x3, x4 = X.T


# Plots:
fig = plt.figure(figsize=(10, 10))
plt.plot(t, x1, 'r-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (day)", fontsize=35)
plt.ylabel(r"$x_1(t)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, x2, 'b-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (day)", fontsize=35)
plt.ylabel(r"$x_2(t)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, x3, 'g-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (day)", fontsize=35)
plt.ylabel(r"$x_3(t)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, x4, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (day)", fontsize=35)
plt.ylabel(r"$x_4(t)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)