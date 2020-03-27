import numpy as np

class GeneticAlgorithm(object):
    '''
    注意此类用于最小化
    初始化参数：
    dna_size            DNA长度，用于表示1个数字的DNA长度，DNA由0，1序列构成
    iteration_num       迭代数
    bound               取值范围，如x1,x2的取值范围[[0,1],[-1,1]]
    evaluator           求值函数，即求最值的函数
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
                 population_size=100,
                 crossover_rate=0.8,
                 mutation_rate=0.01,
                 selection_strategy='roulette_wheel',
                 **kw):
        self.translate_bound(bound)#平移区间
        self.evaluator=evaluator
        kw['selection_strategy']=selection_strategy
        self.fitness={}#用于记录过程
        self.fitness['mean']=[]
        self.fitness['best']=[]
        self.fitness['best_solution']=[]
        self.solution=np.zeros((len(bound),1))
        self.run(iteration_num,dna_size,population_size,crossover_rate,mutation_rate,kw)

    
    def run(self,iteration_num,dna_size,population_size,crossover_rate,mutation_rate,kw):
        '''
        运行整个训练过程初始化->repeat{交叉->变异->选择}
        '''
        population=self.init_population(population_size,dna_size)
        for i in range(iteration_num):
            child=self.mutate(self.crossover(population,crossover_rate),mutation_rate)
            population=self.select(population,kw)

    def mutate(self, population,mutation_rate):
        '''
        变异函数，构建一个bool矩阵1变异，0不变异，即进行异或运算
        '''
        mutation_matrix=np.random.rand(population.shape[0],population.shape[1])<mutation_rate
        mutation_children=population.astype(bool)^mutation_matrix
        return mutation_children

    
    def crossover(self, population,crossover_rate):
        '''
        交叉函数，进行种群数次迭代，随机选择交叉的对象
        '''
        copy_parents=population.copy()
        np.random.shuffle(copy_parents)#打乱复制的数组
        crossover_points=np.random.randint(0,2,size=population.shape).astype(bool)
        crossover_population=np.random.rand(population.shape[0])<crossover_rate
        crossover_points=crossover_points*crossover_population.reshape(population.shape[0],1)
        population[crossover_points]=copy_parents[crossover_points]
        return  population                      

    def translate_dna(self, dna_sequences):
        '''
        将DNA序列转化为数值，第一步将二进制映射到[0,1]，第二步缩放上界倍数并平移回原区间
        '''
        dna_size=dna_sequences.shape[1]/len(self.upper_bound)
        temp=np.tile(2 ** np.arange(dna_size)[::-1],len(self.upper_bound))*dna_sequences
        temp=np.sum(temp.reshape(temp.shape[0],-1,int(dna_size)),axis=2)
        result=temp/ float(2**dna_size-1) * self.upper_bound+self.origin_lower_bound
        return  result#注意返回的数组shape=(种群数，参数数)

    def select(self, offspring,kw):
        '''
        从子代中选择,4种选择策略，通过字符串指定
        '''
        population_size=offspring.shape[0]
        fitness=self.compute_fitness(offspring)
        fitness=np.abs(fitness-np.max(fitness))+1e-13#最大最小转换
        index=None
        if kw['selection_strategy']=='rank':
            fitness_argsort=fitness.argsort()
            fitness_rank=np.array(fitness)
            for i in range(len(fitness_argsort)):
                fitness_rank[fitness_argsort[i]]=i+1.0
            index=np.random.choice(np.arange(population_size), size=population_size, replace=True,p=fitness_rank/sum(fitness_rank))

        elif kw['selection_strategy']=='roulette_wheel':
            index = np.random.choice(np.arange(population_size), size=population_size, replace=True,p=fitness/sum(fitness))

        elif kw['selection_strategy']=='tournament':
            #随机选k个
            index = np.random.choice(np.arange(population_size), size=kw['k'], replace=True)
            select_fitness=[fitness[item] for item in index]
            #在k个中选最优
            index = np.random.choice(np.arange(kw['k']), size=kw['k'], replace=True,p=select_fitness/sum(select_fitness))#此index属于select_fitness
            index=[fitness.index(select_fitness[item]) for item in index]#通过值转化为fitness的index


        elif kw['selection_strategy']=='uniform':#几乎没用
            index = np.random.choice(np.arange(population_size), size=population_size, replace=True)
        
        return offspring[index]

    
    def compute_fitness(self, population):
        '''
        计算适应度,记录每步迭代的结果
        '''
        solutions=self.translate_dna(population)
        fitness=self.evaluator(solutions)
        self.fitness['mean'].append(np.sum(fitness)/fitness.size)
        self.fitness['best'].append(np.min(fitness))
        self.fitness['best_solution'].append(solutions[np.argmin(fitness)])       
        return fitness

    def get_result(self):
        '''
        返回结果
        '''
        return self.fitness
    
    def translate_bound(self,bound):
        '''
        转换区间到[0,n],代入evaluator中时加上原本的下界
        '''
        self.upper_bound=np.zeros((len(bound),1))
        bound=np.array(bound,dtype=float)
        self.origin_lower_bound=bound[:,0].reshape(bound.shape[0],1)
        self.upper_bound=(bound-self.origin_lower_bound)[:,-1].reshape(bound.shape[0],1)
        self.upper_bound=self.upper_bound.T
        self.origin_lower_bound=self.origin_lower_bound.T

    def init_population(self,population_size,dna_size):
        '''
        生成第一代,使用二维数组表示shape=（种群数，参数数量*DNA长度）每DNA长度位表示一个参数
        '''
        return np.random.randint(0,2,size=(population_size,len(self.upper_bound)*dna_size))