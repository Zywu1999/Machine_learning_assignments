import numpy as np


class GeneticAlgorithm(object):
    '''
    初始化参数：
    dna_size            DNA长度，用于表示1个数字的DNA长度，DNA由0，1序列构成
    iteration_num       迭代数
    bound               取值范围如x1,x2的取值范围[[0,1],[-1,1]]
    evaluator          求值函数，即求最值的函数
    population_size     种群大小
    crossover_rate      交叉概率,一般在区间[0.6,0.9]内
    mutation_rate       变异概率
    selection_strategy  选择子代使用的策略,包括roulette_wheel（轮盘选择）,rank,
                        tournament（可指定第一次选择数量k）,uniform
    '''
    def __init__(self,
                 bound:list,
                 iteration_num,
                 evaluator,
                 dna_size=10,
                 population_size=50,
                 crossover_rate=0.7,
                 mutation_rate=0.01,
                 selection_strategy='rank',
                 **kw):
        self.translate_bound(bound)
        self.evaluator=evaluator
        kw['selection_strategy']=selection_strategy
        self.run(iteration_num,dna_size,population_size,crossover_rate,mutation_rate,kw)
        self.fitness={}
        self.fitness['mean']=[]
        self.fitness['best']=[]
        self.fitness['best_solution']=[]
        self.solution=np.zeros((len(bound),1))

    def translate_bound(self,bound):
        '''
        转换区间到[0,n],代入evaluator中时加上原本的下界
        '''
        self.upper_bound=np.zeros((len(bound),1))
        bound=np.array(bound,dtype=float)
        self.origin_lower_bound=bound[:,0].reshape(bound.shape[0],1)
        self.upper_bound=bound-self.origin_lower_bound

    def init_population(self,population_size,dna_size):
        '''
        生成第一代,使用三维数组表示shape=（种群数，参数数量，DNA长度）
        '''
        return np.random.randint(0,2,size=(self.population_size,len(self.upper_bound),dna_size)


    def run(self,iteration_num,dna_size,population_size,crossover_rate,mutation_rate,kw):
        '''
        运行整个训练过程
        '''
        population=self.init_population(population_size,dna_size)
        for i in range(iteration_num):
            child=self.mutate(self.crossover(population,crossover_rate),mutation_rate)
            population=self.select(population,kw)

    def mutate(self, population,mutation_rate):
        '''
        变异函数
        '''
        mutation_matrix=np.random.rand(population.shape[0],population.shape[1],population.shape[2])<mutation_rate
        mutation_children=population.astype(bool)^mutation_matrix
        return mutation_children

    def crossover(self, population,crossover_rate):
        '''
        交叉函数
        '''
        parents=population.reshape(population.shape[0],-1)
        copy_parents=parents.copy()
        for i in range(parents.shape[0]):
            if np.random.rand() < crossover_rate:
                j = np.random.randint(0, parents.shape[0], size=1)                             
                cross_points = np.random.randint(0, 2, size=parents.shape[1]).astype(np.bool)   
                parents[i,:cross_points] = copy_parents[j, :cross_points] 

        crossover_children=parents.reshape(population.shape)
        return  crossover_children                       

    def translate_dna(self, dna_sequence):
        '''
        将DNA序列转化为数值
        '''
        temp=dna_sequence.dot((2 ** np.arange(self.dna_size)[::-1]).reshape(self.dna_size,1))
        result=temp/ float(2**self.dna_size-1) * self.upper_bound+self.origin_lower_bound
        return  result

    def select(self, offspring,kw):
        '''
        从子代中选择
        '''
        population_size=offspring.shape[0]
        fitness=self.compute_fitness(offspring)

    def compute_fitness(self, population):
        '''
        计算适应度,记录每步迭代的结果
        '''
        fitness=[]
        solutions=[]
        for i in range(population.shape[0]):
            solution=self.translate_dna(dna_sequence)
            value=self.evaluator(solution)
            fitness.append(value)
            solutions.append(solution)

        self.fitness['mean'].append(sum(fitness)/len(fitness))
        self.fitness['best'].append(fitness[fitness.index(min(fitness))])
        self.fitness['best_solution'].append(solutions[solutions.index(min(fitness))])       
        return fitness

    def get_result(self):
        '''
        返回结果
        '''
        return self.fitness