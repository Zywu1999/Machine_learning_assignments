
'''
基础的进化策略主要分为三种
(1+1):一个子代一个父代
(μ+λ):从μ个父代与λ个子代中选择μ个最优作为下一代
(μ,λ):λ个子代中选出μ个最优作为下一代，(μ,λ)为主流
'''

import numpy as np 

class EvolutionStrategyBasic(object):
    '''
    基类
    '''
    def __init__(self,population_size,offspring_size,iterations,bound,evaluator):
        '''
        初始化参数：
        population_size     种群数，即μ
        offspring_size      子代数，即λ
        iterations          迭代次数
        bound               取值范围使用二维列表表示[[-1,1],[-1,1]]
        evaluator           评价函数
        '''
        self.result={}#储存信息
        self.result['mean']=[]
        self.result['best']=[]
        self.result['best_solution']=[]
        self.population_size=population_size
        population=self._init_population(bound)
        self._run(population,offspring_size,iterations,evaluator)

    def _run(self,population,offspring_size,iterations,evaluator):
        for i in range(iterations):
            offspring=self._generate_offspring(population,offspring_size)
            population=self._select(offspring,evaluator)

    def _generate_offspring(self,population,offspring_size):
        '''
        生成后代,包括crossover与mutate
        '''
        population=self._crossover(population,offspring_size)
        population=self._mutate(population)
        return population
    
    def _mutate(self,population):
        return population

    def _crossover(self,population,offspring_size):
        return population

    def _select(self,offspring,evaluator):
        '''选择λ个后代'''
        fitness=evaluator(offspring)
        self.result['mean'].append(np.sum(fitness)/fitness.size)
        self.result['best'].append(np.min(fitness))
        self.result['best_solution'].append(offspring[np.argmin(fitness)]) 
        sorted_index=fitness.argsort()[:self.population_size]
        return offspring[sorted_index]

    def _init_population(self,bound):
        '''
        population.shape=(population_size,param_num)
        '''
        bound=np.array(bound)
        param_num=bound.shape[0]
        bound_length=(bound[:,1]-bound[:,0]).reshape(1,param_num)
        self.lower_bound=bound[:,0].reshape(1,param_num)
        self.upper_bound=bound[:,1].reshape(1,param_num)
        population=np.random.rand(self.population_size,param_num)
        population=population*bound_length+self.lower_bound
        return population

    def get_result(self):
        return self.result



class ES_1_1(EvolutionStrategyBasic):
    '''
    (1+1)ES
    '''
    def __init__(self, iterations,bound,evaluator,**kwargs):
        #以下几个参数用于调整变异程度σ       
        self.c=(0.85 if not 'c' in kwargs.keys() else kwargs['c'])
        self.n=2 if  not 'n' in kwargs.keys() else kwargs['n']
        self.t=0 if not 't' in kwargs.keys() else kwargs['t']
        self.record=[]
        self.mutation_strength=0.5 if not 'mutation_strength' in kwargs.keys() else kwargs['mutation_strength']
        #
        super().__init__(1,1,iterations,bound,evaluator)
    
    def _mutate(self,parent):
        offspring=parent+self.mutation_strength*np.random.randn(parent.shape[1])
        offspring=np.clip(offspring,self.lower_bound,self.upper_bound)#保证范围不过线
        return np.concatenate((parent,offspring),axis=0)
    
    def _select(self,offspring,evaluator):
        child=None
        fitness=evaluator(offspring)
        if fitness[0]<fitness[1]:
            child=offspring[0]
            self.record.append(0)
        else:
            child=offspring[1]
            self.record.append(1)
        #调整σ大小，1/5法则
        if self.t%self.n==0:
            ps=sum(self.record[-10*self.n:])/(10*self.n)
            if ps>1.0/5.0:
                self.mutation_strength/=self.c
            elif ps<1.0/5.0:
                self.mutation_strength*=self.c
        
        self.t+=1
        self.result['mean'].append(np.sum(fitness)/fitness.size)
        self.result['best'].append(np.min(fitness))
        self.result['best_solution'].append(offspring[np.argmin(fitness)]) 
        return child.reshape(1,-1)
    
class ES_mu_lambda(EvolutionStrategyBasic):
    '''
    μ+λ与μ,λ
    '''
    def __init__(self,population_size,offspring_size,iterations,bound,evaluator,use_strategy_plus=False):
        #下面为用于调整σ的因子
        self.factor2=1/((2*len(bound))**0.5)
        self.factor1=1/((2*((len(bound))**0.5))**0.5)
        self.use_strategy_plus=use_strategy_plus
        super().__init__(population_size,offspring_size,iterations,bound,evaluator)

    def _init_population(self,bound):
        population={}
        population['value']=super()._init_population(bound)
        population['sigma']=np.random.rand(self.population_size,len(bound))
        assert population['sigma'].shape==population['value'].shape
        return population


    def _mutate(self,population):
        shape=population['sigma'].shape
        random1=np.random.randn(shape[0],1)#每个个体不同
        random2=np.random.randn(1,1)#每个个体相同
        population['sigma']*=np.exp(self.factor1*random1+self.factor2*random2)
        population['value']+=population['sigma']*np.random.randn(*shape)
        population['value']=np.clip(population['value'],self.lower_bound,self.upper_bound)#保证范围不过线
        return population

    def _crossover(self,population:dict,offspring_size):
        #生成λ个子代
        index=np.random.choice(self.population_size,size=offspring_size,replace=True)
        offspring={}
        offspring['sigma']=population['sigma'][index]
        offspring['value']=population['value'][index]
        copy_offspring=offspring.copy()
        #以相同顺序打乱复制的数组的mu与value
        state=np.random.get_state()
        np.random.shuffle(copy_offspring['sigma'])
        np.random.set_state(state)
        np.random.shuffle(copy_offspring['value'])
        #交叉有多种方法，这里随机取两个父代的值
        cp = np.random.randint(0, 2, offspring['sigma'].shape, dtype=np.bool)  
        offspring['sigma'][cp]=offspring['sigma'][cp]
        offspring['sigma'][~cp]=copy_offspring['sigma'][~cp]
        offspring['value'][cp]=offspring['value'][cp]
        offspring['value'][~cp]=copy_offspring['value'][~cp]
        #若使用+策略需合并子代父代
        if self.use_strategy_plus:
            offspring['value']=np.concatenate((offspring['value'],population['value']),axis=0)
            offspring['sigma']=np.concatenate((offspring['sigma'],population['sigma']),axis=0)
        return offspring
    
    def _select(self,offspring,evaluator):
        fitness=evaluator(offspring['value'])
        self.result['mean'].append(np.sum(fitness)/fitness.size)
        self.result['best'].append(np.min(fitness))
        self.result['best_solution'].append(offspring['value'][np.argmin(fitness)])
        self.result['sigma']=np.mean(offspring['sigma'],axis=0)
        sorted_index=fitness.argsort()[:self.population_size]
        offspring['sigma']=offspring['sigma'][sorted_index]
        offspring['value']=offspring['value'][sorted_index]
        return offspring



def main():
    D=2
    bound=[-100,100]
    bounds=[bound for i in range(D)]
    model=ES_mu_lambda(population_size=50,offspring_size=500,iterations=5000,bound=bounds,evaluator=easom)
    print('best fitness=',model.result['best'][-1],'\n best_solution=',model.result['best_solution'][-1])
    print('sigma',model.result['sigma'])
    from matplotlib import pyplot as plt
    ax1=plt.subplot(111)
    ax1.plot(np.arange(len(model.result['best'])),model.result['best'],'b')
    ax1.plot(np.arange(len(model.result['best'])),model.result['mean'],'r')
    plt.show()

if __name__ == "__main__":
    from test_functions import *
    main()