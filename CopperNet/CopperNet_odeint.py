"""
Implementation of CopperNet, the compartmental network 
proposed in [1] for controlling the flow of copper in a 
supply-recovery chain. The network architecture is inspired
by the study in [2].
This code is part of the source code of [1].

References:
    [1] Zocco, F., Corti, A. and Malvezzi, M., 2025. CiRL: 
        Open-source environments for reinforcement learning 
        in circular economy and net zero. arXiv preprint 
        arXiv:2505.21536. 
    [2] Loibl, A. and Espinoza, L.A.T., 2021. Current 
        challenges in copper recycling: Aligning insights 
        from material flow analysis with technological 
        research developments and industry issues in Europe 
        and North America. Resources, Conservation and 
        Recycling, 169, p.105462.
"""


import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#######################Simulator settings#####################
# Simulation time:
t_final = 150


# Model parameters:
# For eq. 1:
a_17 = 0.72 # [1/day]
a_16 = 0.11 # [1/day]
a_21 = 0.82 # [1/day]
# For eq. 2:
a_26 = 0.19 # [1/day]
a_32 = 1.0 # [1/day]
# For eq. 3:
a_43 = 0.84 # [1/day] 
# For eq. 4:
a_84 = 0.25 # [1/day]
# For eq. 5:
a_65 = 0.19 # [1/day]
# For eq. 6:
# no further parameters


# Initial conditions:
x1_ini = 65.0 # [kt]
x2_ini = 80.0 # [kt]
x3_ini = 80.0 # [kt]
x4_ini = 1300.0 # [kt]
x5_ini = 15.0 # [kt]
x6_ini = 15.0 # [kt]
X_ini = np.array([x1_ini, x2_ini, x3_ini, x4_ini, x5_ini, x6_ini])


# Inputs: 
u = 1 # arbitrary choice for code testing


def CopperNet(X, t=0):
    
    return np.array([a_17*X[0] + a_16*X[5] - a_21*X[0],
                     a_21*X[0] + a_26*X[5] - a_32*X[1],
                     a_32*X[1] - a_43*X[2],
                     a_43*X[2] - a_84*X[3] - u,
                     -a_65*X[4] + u,
                     a_65*X[4] - a_16*X[5] - a_26*X[5]])


# Numerical solution:
t = np.linspace(0, t_final, 1000)
X, infodict = integrate.odeint(CopperNet, X_ini, t,
mxstep = 1000, full_output = True)
x1, x2, x3, x4, x5, x6 = X.T


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

fig = plt.figure(figsize=(10, 10))
plt.plot(t, x5, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (day)", fontsize=35)
plt.ylabel(r"$x_5(t)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, x6, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (day)", fontsize=35)
plt.ylabel(r"$x_6(t)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)