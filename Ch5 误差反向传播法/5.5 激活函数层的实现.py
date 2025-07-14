import numpy as np
class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0)      #mask变量保存了由True/False构成的Numpy数组
        out=x.copy()
        out[self.mask]=0        #输入x的元素中小于等于0的地方都变成了0
        return out
    def backward(self,dout):
        dout[self.mask]=0       #若正向传播时的x<=0，反向传播给下游的信号为0
        dx=dout                 #若正向传播时的x>0,反向传播会将上游的值原封不动传给下游
        return dx
    
class Sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out            #正向传播时将输出保存在实例变量out中，反向传播时会用
        return out
    def backward(self,dout):
        dx=dout*self.out*(1-self.out)
        return dx