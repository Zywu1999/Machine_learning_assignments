from genetic_algorithm import GeneticAlgorithm
import numpy as np
from test_functions import *
bound=[-100,100]
d=2
bounds=[bound for i in range(d)]
model=GeneticAlgorithm(bound=bounds,iteration_num=5000,dna_size=15,
                      evaluator=easom,mutation_rate=0.01,population_size=100,selection_strategy='rank')
min_fitness=model.fitness['best'][-1]
print('best fitness=',min_fitness,'\n best_solution=',model.fitness['best_solution'][-1])

from matplotlib import pyplot as plt
ax1=plt.subplot(111)
ax1.plot(np.arange(len(model.fitness['best'])),model.fitness['best'],'b')
ax1.plot(np.arange(len(model.fitness['best'])),model.fitness['mean'],'r')
plt.show()