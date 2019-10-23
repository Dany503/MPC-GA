# Libraries importation
import numpy as np
from numba import njit

# Power profiles
PV = np.array([0, 0, 0, 0, 0, 0, 0, 0, 6, 10, 15, 20, 30, 40, 40, 20, 15, 10, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 10, 15, 20, 30, 40, 40, 20, 15, 10, 8, 2, 0, 0, 0, 0], dtype = float)
WT = np.array([51, 51, 58, 51, 64, 51, 44, 51, 44, 51, 51, 46, 81, 74, 65, 65, 65, 51, 39, 63, 38, 66, 74, 74, 51, 51, 58, 51, 64, 51, 44, 51, 44, 51, 51, 46, 81, 74, 65, 65, 65, 51, 39, 63, 38, 66, 74, 74], dtype = float)
DM = np.array([67, 67, 90, 114, 120, 130, 150, 190, 200, 206, 227, 227, 250, 250, 200, 180, 160, 160, 190, 150, 100, 50, 20, 20, 67, 67, 90, 114, 120, 130, 150, 190, 200, 206, 227, 227, 250, 250, 200, 180, 160, 160, 190, 150, 100, 50, 20, 20], dtype = float)
#PV_noise = np.random.normal(0, 4, 48)
PV_noise = np.array([-3.5820185 , -2.20364537,  0.91936474, -5.99991692, -6.3390374 ,
       -1.41080227,  0.40519591, -8.91915404,  2.71505025, -0.11925967,
        4.22982719,  3.95832155, -2.71136049, -7.24385201,  2.12597179,
        1.81870281,  1.21692113, -2.37740575,  3.33809536, -0.6038406 ,
       -8.84318258, -5.63491849,  4.39498836, -4.78524406,  6.6708922 ,
        4.47062287, -1.69095277, -1.08331732,  5.3901347 ,  3.13104228,
        7.98573166, -1.17464752,  2.2910306 , -6.43775553,  3.01456996,
        5.0482637 ,  0.32688031, -5.83141882,  2.05162122,  3.28465431,
        1.372145  , -0.86209194,  0.60492186, -3.24859197,  4.54036528,
       -4.56340252,  3.04978166, -1.97432668])
#WT_noise = np.random.normal(0, 4, 48)
WT_noise = np.array([ 5.20537738,  0.87628391,  1.14343065,  2.62149959, -4.33157711,
       -5.60322081,  3.68933275, -1.45236942, -8.67238249, -2.1967041 ,
        4.91949607,  5.6653824 ,  1.90293758,  3.55941269,  8.32153686,
        1.61998372, -0.06951529,  0.88559032, -3.89284542,  2.07581318,
        3.82620176,  1.94761344,  5.20496953,  2.79118514, -1.40866512,
        5.09973216,  0.58736119, -0.50072147,  4.98383048,  0.96778639,
       11.83735369,  3.41393102,  0.61878416, -4.02894608,  6.45991416,
       -1.00782133, -0.95165031, -2.90279602,  2.91749408, -4.35157266,
       -6.88668281,  2.21569665, -1.94214263,  2.09337459, -4.62966945,
        2.14836656,  1.9538667 ,  5.05385272])
#DM_noise = np.random.normal(0, 2.5, 48)
DM_noise = np.array([-4.10870404,  1.50938494, -7.70210856, -4.77043582,  4.79836324,
        6.22113061,  2.11573627, -2.9165049 , -2.81958772, -1.38893633,
        1.11185661, -8.16492539, -2.01799609, -5.61744562, -1.34089347,
        4.09619259, -0.38322569, -1.98890466,  1.32332414, -1.2828731 ,
       -3.22543985, -7.02675216,  2.29867403,  1.29761234,  6.64760441,
        4.54718301, -1.70827244, -3.13666248, -5.82160464, -0.83092904,
       -3.47414209,  8.27087094, -3.14158869, -0.95793074,  1.8948983 ,
       -2.45902192,  3.75412138, -6.6773956 ,  3.87436162, -4.58220922,
       -4.20165589,  8.16249627, -1.00500049, -4.29285434,  2.23526952,
       -3.19472556, -5.21525579, -1.75813815])
P_dem = DM - PV - WT

# Diesel engine
P_DE_min = 5
P_DE_max = 80
a_DE = 0.3
b_DE = 0.4
c_DE = 5.2
d_DE = 1.925
e_DE = 0.2455
f_DE = 0.0012
OM_DE = 0.01258
T_DE = 0

# Microturbine 
P_MT_min = 20
P_MT_max = 140
a_MT = 0.4
b_MT = 0.28
c_MT = 7.1
d_MT = 7.4344
e_MT = 0.2015
f_MT = 0.0002
OM_MT = 0.00587
T_MT = 0

# Battery bank 
P_BB_min = -120
P_BB_max = 120
eta_c = 0.9
eta_d = 0.9
SOC_min = 70
SOC_max = 280
SOC_ini = (SOC_max + SOC_min)/2

# MPC paramters
H = 10
t_now = 0

# GA parameters
pen = 1000000000


def initialize():
    P_dem[0] += DM_noise[0] - PV_noise[0] - WT_noise[0]
    return SOC_ini, 0, 0, 0, P_dem 

# CONSTRAINTS
# Power limits 
@njit
def P_limits(P, Pmin, Pmax):
    aux = np.zeros(H)
    for index in range(H):
        if P[index] != 0:
            aux[index] = P[index] - Pmin
    return np.concatenate((aux, Pmax- P))
# State of charge computation and limits
@njit
def SOC_evolution(SOC, P, Dt):
    SOC_ = np.zeros(H+1)
    SOC_[0] = SOC
    for index in range(H):        
        if P[index] < 0:
            SOC_[index+1] = SOC_[index] - P[index]*Dt*eta_c
        else:
            SOC_[index+1] = SOC_[index] - P[index]*Dt/eta_d
    return SOC_[1:]
@njit
def SOC_limits(SOC, SOCmin, SOCmax):
    return np.concatenate((SOC - SOCmin, SOCmax - SOC))

# COST TERMS
# Operation and mainteinance cost
@njit
def OM_cost(k, P, Dt):
    return k*np.sum(P)*Dt
# DE operation and start-up cost
@njit
def cost_DE_t(P,T0):
    t = T0
    cost = 0
    for p in P:
        if (p > 0):
            if p < 0.1:
                cost += d_DE
                t = 0
            else:
                if t == 0:
                    cost += f_DE*(p**2) + e_DE*p + d_DE
                else:
                    cost += (f_DE*(p**2) + e_DE*p + d_DE) + (a_DE + b_DE*(1-np.exp(-t/c_DE)))
                t = 0
        else:
            cost += 0
            t += 1
    return cost
# MT operation and start-up cost
@njit
def cost_MT_t(P,T0):
    t = T0
    cost = 0
    for p in P:
        if (p > 0):
            if p < 0.1:
                cost += d_MT
                t = 0
            else:
                if t == 0:
                    cost += f_MT*(p**2) + e_MT*p + d_MT
                else:
                    cost += (f_MT*(p**2) + e_MT*p + d_MT) + (a_MT + b_MT*(1-np.exp(-t/c_MT)))
                t = 0
        else:
            cost += 0
            t += 1
    return cost

# NEXT TIME INSTANT
def next_step(x, iSOC_actual, iT_DE, iT_MT, iP_dem, it_now):
    P_DE = x[:H]
    P_MT = x[H:]
    if P_DE[0] != 0:
        T_DE = 0
    else:
        T_DE = iT_DE + 1
    if P_MT[0] != 0:
        T_MT = 0
    else:
        T_MT = iT_MT + 1
    P_bat = iP_dem[it_now] - P_DE[0] - P_MT[0]
    if P_bat < 0:
        SOC = iSOC_actual - P_bat*eta_c
    else:
        SOC = iSOC_actual - P_bat/eta_d
    t_now = it_now + 1
    P_dem[t_now] += DM_noise[t_now] - PV_noise[t_now] - WT_noise[t_now]
    return SOC, T_DE, T_MT, P_dem, t_now
    
# x = [P_DE (x H), P_MT (x H)]
# The power of the battery is obtained from the rest of generators
@njit
def fitness(x, SOC_actual, T_DE, T_MT, P_dem, t_now):
    P_DE = x[:H]
    P_MT = x[H:]
    P_BB = P_dem[t_now:t_now + H] - P_DE - P_MT
    P_DE_lim = P_limits(P_DE, P_DE_min, P_DE_max)
    P_MT_lim = P_limits(P_MT, P_MT_min, P_MT_max)
    P_BB_lim = P_limits(P_BB, P_BB_min, P_BB_max)
    for index in range(H):
        if P_DE_lim[index] < 0:
            return pen+1,
        if P_MT_lim[index] < 0:
            return pen+2,
        if P_BB_lim[index] < 0:
            return pen+3,
    SOC = SOC_evolution(SOC_actual, P_BB, 1)
    SOC_lim = SOC_limits(SOC, SOC_min, SOC_max)
    for index in range(H):
        if SOC_lim[index] < 0:
            return pen+4,
    cost = OM_cost(OM_DE, P_DE, 1)
    cost += OM_cost(OM_MT, P_MT, 1)
    cost += cost_DE_t(P_DE, T_DE)
    cost += cost_DE_t(P_MT, T_MT)
    return cost,

def fitness_res(x, SOC_actual, T_DE, T_MT, P_dem):
    P_DE = x[:24]
    P_MT = x[24:] 
    cost = OM_cost(OM_DE, P_DE, 1)
    cost += OM_cost(OM_MT, P_MT, 1)
    cost += cost_DE_t(P_DE, T_DE)
    cost += cost_DE_t(P_MT, T_MT)
    return cost,

# Creation of new individuals
@njit
def create_ind():
    ind = np.zeros(2*H)
    for i in range(0, H):
        ind[i] = np.random.uniform(P_DE_min, P_DE_max)
#        ind[H + i] = P_dem[i] - ind[i]
        ind[H + i] = np.random.uniform(P_MT_min, P_MT_max)
        if ind[H + 1] < 0:
            ind[H + 1] = 0
        if ind[H + i] < P_MT_min:
            ind[H + i] = 0
        if ind[H + i] > P_MT_max:
            ind[H + i] = P_MT_max  
    return ind

# Mutation
@njit
def mutation(ind, indpb):
    for j, i in enumerate(ind):
        if np.random.random() < indpb[0]:
            ind[j] = np.random.normal(ind[j], 30)
        if np.random.random() < indpb[1]:
            ind[j] = 0
        if np.random.random() < indpb[2]:
            ind[j] = 0.005
    return ind,

def eval_solution(x):
    x = np.array(x)
    P_DE = x[:24]
    P_MT = x[24:]
    cost = OM_cost(OM_DE, P_DE, 1)
    cost += OM_cost(OM_MT, P_MT, 1)
    cost += cost_DE_t(P_DE, 0)
    cost += cost_DE_t(P_MT, 0)
    return cost

