""" 
Implementation of the Droop's model solved via integrate.odeint() to tune 
the value of the integration step 'dt' of the Euler's method. Then, the 
value of 'dt' is used in 'Droop_createAndTestEnvironment.ipynb' for
numerical integration. 
This code is part of the source code of [1]. The Droop's model is 
taken from [2,3,4]. 

References:
    [1] Zocco, F., Corti, A. and Malvezzi, M., 2025. CiRL: Open-source environments for reinforcement learning in circular economy and 
    net zero. arXiv preprint arXiv:2505.21536.
    [2] Vatcheva, I., De Jong, H., Bernard, O. and Mars, N.J., 2006. Experiment 
    selection for the discrimination of semi-quantitative models of dynamical systems. 
    Artificial Intelligence, 170(4-5), pp.472-506.
    [3] Bernard, O., 2011. Hurdles and challenges for modelling and control of 
    microalgae for CO2 mitigation and biofuel production. Journal of Process 
    Control, 21(10), pp.1378-1389.
    [4] Zocco, F., Garcia, J. and Haddad, W.M., 2025. Circular microalgae-based
    carbon control for net zero. arXiv preprint arXiv:2502.02382.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#######################Simulator settings#####################
# Simulation time:
t_final = 15

# Model parameters (from Fig. 8 in [2] and from [3]):
K_sI = 0 # value not found, so at the moment it is set as 0
K_iI = 295 # from [3]
mu_tilde = 1.7 # from table 1 in [3]
k_q = 1.7 # from [2]
K_S = 0.1 # from Fig. 8 in [2]
rho_max = 9.40
T_h = 1/0.45 # it is 1/D, with D taken from Table 1 in [2]
S_in = 100  

# Model parameter taken from [4]: 
K_CO2 = 0.3 # 0 < K_CO2 < 1; value chosen since not found



# Initial conditions:
I_bar = 50 # value chosen as a "frequent" value in Fig. 7 of [3]
X_ALG_ini = 26 # from Table 1 in [2]
Q_ini = 2.82 # from Table 1 in [2]
S_ini = 0 # from Table 1 in [2] 
X_ini = np.array([X_ALG_ini, Q_ini, S_ini])  

# Inputs:
I = I_bar
################################################################

# Equations in state space form:
def microalgaeDroop(X, t=0): 
    mu = mu_tilde*(I/(I+K_sI+(I**2/K_iI)))*(1-k_q/X[1])
    rho = rho_max*(X[2]/(X[2]+K_S)) 
    
    return np.array([mu*X[0] - (1/T_h)*X[0],
        rho - mu*X[1],
        (1/T_h)*(S_in - X[2]) - rho*X[0]])

# Numerical solution:
t = np.linspace(0, t_final, 1000)
X, infodict = integrate.odeint(microalgaeDroop, X_ini, t,
mxstep = 1000, full_output = True)
X_ALG, Q, S = X.T


# Absorption of CO2:
rho = rho_max*(S/(S+K_S))
m_dot23 = K_CO2*rho # i.e., rho_CO2
    

# Plots
fig = plt.figure(figsize=(10, 10))
plt.plot(t, X_ALG, 'r-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Algal biomass, $X_{ALG} \, \left(\frac{\mu m^3}{L}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, Q, 'b-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Internal cell quota, $Q \, \left(\frac{\mu mol}{\mu m^3}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, S, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Remaining nutrients, $S \, \left(\frac{\mu mol}{L}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, m_dot23, 'b-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"$CO_2$ flow, $\dot{m}_{2,3} \, \left(\frac{\mu mol}{\mu m^3d}\right)$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

# For debugging purposes, S+XQ should behave as a first order:
fig = plt.figure(figsize=(10, 10))
plt.plot(t, S+X_ALG*Q, 'r-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (d)", fontsize=35)
plt.ylabel(r"Debugging variable, $S+X_{ALG} Q$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)