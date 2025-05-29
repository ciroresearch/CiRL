"""
Implementation of the incinerator compartment. This code is 
part of the source code of [1].

This model is a simplified version of that developed in [2].
The values of the parameters are not taken from a
real incinerator, and hence, they are merely to demonstrate
the use of reinforcement learning. This model is 
described by mass and energy balances [1,2], and hence, the 
incinerator is a compartment of a thermodynamical material 
network [3].

References:
    [1] Zocco, F., Corti, A. and Malvezzi, M., 2025. CiRL: Open-source environments for reinforcement learning in circular economy and 
    net zero. arXiv preprint arXiv:2505.21536.
    [2] Magnanelli, E., Tranås, O.L., Carlsson, P., Mosby, 
    J. and Becidan, M., 2020. Dynamic modeling of municipal 
    solid waste incineration. Energy, 209, p.118426.
    [3] Zocco, F., Sopasakis, P., Smyth, B. and Haddad,
    W.M., 2023. Thermodynamical material networks for 
    modeling, planning, and control of circular 
    material flows. International Journal of 
    Sustainable Engineering, 16(1), pp.1-14.
"""

import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#######################Simulator settings#####################
# Simulation time:
t_final = 3*60 #minutes


# Model parameters:
# For eq. 1:
F_in = 100 #inlet waste
F_out = 10 #outlet ashes
R_w = 70 #consumption rate
# For eq. 2:
F_char_out = 2 #output flow of char 
P_char = 6 #production rate due to pyrolysis
R_char = 3 #consumption rate due to combustion
# For eq. 3:
c_p_w = 1
F_aI = 5
c_p_g = 5
T_aI = 20 #ambient temperature
Q = 100 #exothermic reaction
c_p_char = 1
c_p_m = 25
M_grate = 500
# For eq. 4:
F_g_w_out = 2 
R_g = 1 #production rate of gases due to combustion/pyrolysis   
# For eq. 5:
F_g_out = 2
F_aII = 3
# For eq. 6:
c_p_m = 5
M_f_b = 120
T_aII = 20
Q_g = 450
    

# Initial conditions:
M_ini = 0
M_char_ini = 0 
T_w_ini = 20
M_g_w_ini = 0
M_g_f_ini = 0
T_g_ini = 20 #ambient temperature
X_ini = np.array([M_ini, M_char_ini, T_w_ini, 
                  M_g_w_ini, M_g_f_ini, T_g_ini])
 
# Inputs:
Q_ext = 40 #heat affecting the temperature of gases in freeboard, T_g 
################################################################

# Equations in state space form:
def incinerator(X, t=0):
        
        return np.array([F_in-F_out-R_w,
                        -F_char_out+P_char-R_char,                         
                         (F_in*c_p_w*X[2]+F_aI*c_p_g*(T_aI-X[2])+Q)/(c_p_w*X[0]+c_p_char*X[1]+c_p_m*M_grate),
                         F_aI-F_g_w_out+R_g,
                         F_g_w_out-F_g_out+F_aII,
                         (F_g_w_out*c_p_g*(X[2]-X[5])+F_aII*c_p_g*(T_aII-X[5])+Q_g-Q_ext)/(c_p_g*X[4]+c_p_m*M_f_b)])
    
# Numerical solution:
t = np.linspace(0, t_final, 1000)
X, infodict = integrate.odeint(incinerator, X_ini, t,
mxstep = 1000, full_output = True)
waste_mass, char_mass, wastebed_temp, gas_mass_wastebed, gas_mass_freeboard, gas_temp_freeboard = X.T

# Plots
fig = plt.figure(figsize=(10, 10))
plt.plot(t, waste_mass, 'r-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (min)", fontsize=35)
plt.ylabel(r"Waste mass, $M$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, char_mass, 'b-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (min)", fontsize=35)
plt.ylabel(r"Char mass, $M_{char}$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, wastebed_temp, 'g-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (min)", fontsize=35)
plt.ylabel(r"Wastebed temperature, $T_w$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, gas_mass_wastebed, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (min)", fontsize=35)
plt.ylabel(r"Gas mass in wastebed, $M_{g,w}$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, gas_mass_freeboard, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (min)", fontsize=35)
plt.ylabel(r"Gas mass in freeboard, $M_{g,f}$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)

fig = plt.figure(figsize=(10, 10))
plt.plot(t, gas_temp_freeboard, 'k-', linewidth=6)
plt.grid()
plt.xlabel(r"Time, $t$ (min)", fontsize=35)
plt.ylabel(r"Gas temperature in freeboard, $T_g$", fontsize=35) 
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)