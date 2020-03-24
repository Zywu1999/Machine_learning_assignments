from genetic_algorithm import GeneticAlgorithm
import numpy as np
def evaluator(x):
    return np.dot(x.T,x)
bound=[-1,1]
d=2
bounds=[bound for i in range(d)]
model=GeneticAlgorithm(bound=bounds,iteration_num=5000,evaluator=evaluator,mutation_rate=0.1,population_size=100,selection_strategy='tournament',k=50)
print('best fitness=',model.fitness['best'][-1],'\n best_solution=',model.fitness['best_solution'][-1])

from matplotlib import pyplot as plt
plt.plot(np.arange(len(model.fitness['best'])),model.fitness['best'],'b')
plt.plot(np.arange(len(model.fitness['best'])),model.fitness['mean'],'r')
plt.show()