import numpy as np
from commonpackage.functions import softmax,cross_entropy_error
class Affine:          #实现神经网络中的仿射变换层
    def __init__(self,w,b):   #初始化成员变量
        self.w=w
        self.b=b              #模型参数，初始化时固定，通过梯度更新
        self.x=None
        self.dw=None
        self.db=None
    def forward(self,x):
        self.x=x
        out=np.dot(x,self.w)+self.b   #这里的x是局部参数，仅在方法内部有效，可以不用self调用
        return out 
    def backward(self,dout):    #dout通常是损失函数对当前层输出的梯度
        dx=np.dot(dout,self.w.T)  #中间梯度，仅在当前反向传播调用中需要，作为局部变量计算后直接返回，因此不是成员变量
        self.dw=np.dot(self.x.T,dout)  
        self.db=np.sum(dout,axis=0)      #dw和db用于后续参数更新，需要累积或保存
        return dx

class SoftmaxWithLoss:
    #两个作用：1.Softmax激活，将网络输出转换为概率分布
    #         2.Loss计算
    def __init__(self):
        self.loss=None      #损失函数值
        self.y=None         #softmax输出
        self.t=None         #监督（标签）数据t
    def forward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=cross_entropy_error(self.y,self.t)
        return self.loss
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        dx=(self.y-self.t)/batch_size     #平均每个样本的梯度
        return dx
