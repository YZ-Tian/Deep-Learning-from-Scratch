import sys,os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from commonpackage.functions import softmax,cross_entropy_error
from commonpackage.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.w=np.random.randn(2,3)  #用正态分布初始化权重，生成两行三列的矩阵
    def predict(self,x):
        return np.dot(x,self.w)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        return loss

net=simpleNet()
# print(net.w)
x=np.array([0.6,0.9])
p=net.predict(x)
t=np.array([0,0,1])
# print(p)
def f(W):
    return net.loss(x,t)
dW=numerical_gradient(f,net.w)     #神经网络损失函数关于权重的梯度
print(dW)