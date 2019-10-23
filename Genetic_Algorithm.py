import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
import random
#from scoop import futures
import time
import MPC
import copy

SOC, T_DE, T_MT, t_now, P_dem = MPC.initialize()

creator.create("Problema1", base.Fitness, weights=(-1,))
creator.create("Individual", np.ndarray, fitness = creator.Problema1)
 
toolbox = base.Toolbox() 
toolbox.register("individual", tools.initIterate, creator.Individual, MPC.create_ind)
toolbox.register("ini_poblacion", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", MPC.fitness, SOC_actual = SOC, T_DE = T_DE, T_MT = T_MT, P_dem = P_dem, t_now = t_now)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", MPC.mutation, indpb=(0.05,0.05, 0.005))
toolbox.register("select", tools.selTournament, tournsize = 3)
#toolbox.register("map", futures.map)

def plot(log, t):
    gen = log.select("gen")
    fit_mins = list(log.select("min"))
    #fit_maxs = list(log.select("max"))
    #fit_ave = list(log.select("avg"))
    
    fig, ax1 = plt.subplots()
    ax1.plot(gen, fit_mins, "b")
    #ax1.plot(gen, fit_maxs, "r")
    #ax1.plot(gen, fit_ave, "--k")
    #ax1.fill_between(gen, fit_mins, fit_maxs, where=fit_maxs >= fit_mins, facecolor='g', alpha = 0.2)
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness")
    #ax1.legend(["Min", "Max", "Avg"])
    ax1.set_ylim([min(fit_mins)-50, min(fit_mins)+100])
    plt.grid(True)
    plt.title("Evolution of best at " + str(t) + " step")
    plt.savefig("Evolution.png")
    plt.show()
    return fit_mins
                
def unico_objetivo_ga(c, m):
    """ los parámetros de entrada son la probabilidad de cruce, la probabilidad
    de mutación y el número iteración
    """
    NGEN = 200 # aumentar a 1000
    MU = 400 # aumentar a 3000
    LAMBDA = MU 
    CXPB = c
    MUTPB = m
    #random.seed(i) # actualizamos la semilla cada vez que hacemos una simulación
   
    pop = toolbox.ini_poblacion(n = MU)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    #hof = tools.HallOfFame(1)
 
    stats = tools.Statistics(key = lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
   
    logbook = tools.Logbook()
   
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN,
                              stats= stats, halloffame=hof, verbose = False)
   
    return pop, hof, logbook


#ind = toolbox.individual()
#print(toolbox.evaluate(ind))

#%%

if __name__ == "__main__":
    random.seed(1)    
    parameters= [(0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
    t0 = time.time()
#    for iteraciones in range(0, 20):
    for iteraciones in range(0, 1):
        solucion_final = [0 for i in range(48)]
        if iteraciones != 0:
            SOC, T_DE, T_MT, t_now, P_dem = MPC.initialize()
        for t in range(0, 24):
            best_list= list()
            c_list = list()
            m_list = list()
            res_individuos = open("individuos.txt", "a")
            res_fitness = open("fitness.txt", "a")
            for c, m in parameters:
                for k in range(5):                    
                    pop_new, best, log = unico_objetivo_ga(c, m)
                    best_list.append(copy.deepcopy(best))
                    c_list.append(c)
                    m_list.append(m)
                    del(pop_new)
                    del(best)
                    
            pareto_new = min(best_list, key = lambda x: x[0].fitness.values[0])
            c = c_list[best_list.index(pareto_new)]
            m = m_list[best_list.index(pareto_new)]
            solucion_final[t] = pareto_new[0][0]
            solucion_final[t+24] = pareto_new[0][10]
            del(best_list)
            del(m_list)
            del(c_list)
            
            for ide, ind in enumerate(pareto_new):
                res_individuos.write(str(t))
                res_individuos.write(",")
                res_individuos.write(str(list(ind)))
                res_individuos.write("\n")
                res_fitness.write(str(t))
                res_fitness.write(",")
                res_fitness.write(str(c))
                res_fitness.write(",")
                res_fitness.write(str(m))
                res_fitness.write(",")
                res_fitness.write(str(ind.fitness.values[0]))
                res_fitness.write("\n")                
                aux = P_dem
                SOC, T_DE, T_MT, P_dem, t_now = MPC.next_step(ind, SOC, T_DE, T_MT, P_dem, t_now)
                toolbox.register("evaluate", MPC.fitness, SOC_actual = SOC, T_DE = T_DE, T_MT = T_MT, P_dem = P_dem, t_now = t_now)
            #del(pop_new)
            del(pareto_new)
            res_fitness.close()
            res_individuos.close()
            print('t = ' + str(t_now) + ', SOC = ' + str(SOC) + ', P_dem[t_now] = ' + str(aux[t_now - 1]))
            
        control_final = open("control_final.txt", "a")
        fitnes_final = open("fitness_final.txt", "a")
        control_final.write(str(iteraciones))
        control_final.write(",")
        control_final.write(str(solucion_final))
        control_final.write("\n")
        fitnes_final.write(str(iteraciones))
        fitnes_final.write(",")
        fitnes_final.write(str(MPC.eval_solution(solucion_final)))
        fitnes_final.write("\n")
        control_final.close()
        fitnes_final.close()
        t1 = time.time()
        print("Duracion %f" %(t1-t0))
            
