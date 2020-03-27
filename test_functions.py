import numpy as np 

'''
外部传的x的shape=(种群数,参数数)
'''
def sphere(x):
    '''
    x_i [-100,100]
    '''
    return np.sum(x*x,axis=1)

def sum_squares(x):
    '''
    x_i [-10,10]
    '''
    i=(np.arange(x.shape[1])+1).reshape(1,x.shape[1])
    return np.sum(i*x*x,axis=1)

def step(x):
    '''
    x_i [-100,100]
    '''
    return np.sum((x+0.5).astype(int)*(x+0.5).astype(int),axis=1)

def quartic(x):
    '''
    x_i [-1.28,1.28]
    '''
    i=(np.arange(x.shape[1])+1).reshape(1,x.shape[1])
    return np.sum(i*x*x*x*x,axis=1)+np.random.rand()

def easom(x):
    '''
    x_i [-100,100]
    '''
    x1,x2=x[:,0],x[:,1]
    return -np.cos(x1)*np.cos(x2)*np.exp(-(x1-np.pi)**2-(x2-np.pi)**2)

def matyas(x):
    '''
    x_i [-10,10]
    '''
    x1,x2=x[:,0],x[:,1]
    return 0.26*(x1**2+x2**2)-0.48*x1*x2
