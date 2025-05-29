"""
Implementation of the transport truck that moves 
the unsorted material "m_u" from the waste sorter 
to the incinerator. This code is part of the source 
code of [1].

The Lagrange's equations are a derivation of the 
first law of thermodynamics as demonstrated in [2].
Hence, a mechanical system whose dynamics is derived 
from the Lagrange's equations, e.g., this truck model,
is a compartment of a thermodynamical material 
network [2,3].

References:
    [1] Zocco, F., Corti, A. and Malvezzi, M., 2025. CiRL: Open-source environments for reinforcement learning in circular economy and 
    net zero. arXiv preprint arXiv:2505.21536.
    [2] Zocco, F., Sopasakis, P., Smyth, B. and Haddad,
    W.M., 2023. Thermodynamical material networks for 
    modeling, planning, and control of circular 
    material flows. International Journal of 
    Sustainable Engineering, 16(1), pp.1-14.
    [3] Zocco, F. and Malvezzi, M., 2025. Circular economy 
    design through system dynamics modeling. In International 
    Workshop IFToMM for Sustainable Development Goals 
    (pp. 530-538). Springer, Cham.
    
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#######################Simulator settings#####################
# Simulation time:
t_final = 15


# Model parameters:
m_truck = 3000 # kg
m_u = 210 # kg
m_total = m_truck + m_u 
    
# Initial conditions:
position_ini = 0
speed_ini = 4
X_ini = np.array([position_ini, speed_ini])
 
# Inputs:
F = 4000 # N 
################################################################

# Equations in state space form:
def transportTruck(X, t=0):
        
        return np.array([X[1],
                         F/m_total])
    
# Numerical solution:
t = np.linspace(0, t_final, 1000)
X, infodict = integrate.odeint(transportTruck, X_ini, t,
mxstep = 1000, full_output = True)
position, speed = X.T

# Plots
fig = plt.figure(figsize=(10, 10))
plt.plot(t, position, 'r-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (s)", fontsize=35)
plt.ylabel(r"Position, $x$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, speed, 'b-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (s)", fontsize=35)
plt.ylabel(r"Speed, $\dot{x}$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)