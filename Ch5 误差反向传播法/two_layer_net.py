# coding: utf-8
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from commonpackage.layers  import *
from commonpackage.gradient import numerical_gradient
from collections import OrderedDict   #从标准库的 collections 模块中导入 OrderedDict 这个类
#OrderedDict 是一种特殊的字典（dict），它与普通字典的主要区别在于会记录键值对的插入顺序
#  OrderedDict 可以明确保留插入时的顺序，便于后续按插入顺序遍历或操作数据。
from dataset.mnist import load_mnist

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        #生成层
        self.layers= OrderedDict()
        self.layers["Affine1"]=Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])
        self.lastLayers=SoftmaxWithLoss()

    def predict(self, x):                 #进行识别（推理）
        # W1, W2 = self.params['W1'], self.params['W2']
        # b1, b2 = self.params['b1'], self.params['b2']
    
        # a1 = np.dot(x, W1) + b1
        # z1 = sigmoid(a1)
        # a2 = np.dot(z1, W2) + b2
        # y = softmax(a2)
        
        # return y
        for layer in self.layers.values():
            #依次遍历Affine、Relu等实例
            x=layer.forward(x)
        return x
            
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):         #数值微分法计算权重参数的梯度
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
        
    def gradient(self, x, t):                  #误差反向传播法计算权重参数的梯度
        #forward
        self.loss(x,t)

        #backward
        dout=1
        dout=self.lastLayers.backward(dout)

        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        
        #收集梯度
        grads={}
        grads['w1']=self.layers['Affine1'].dW
        grads['b1']=self.layers['Affine1'].db
        grads['w2']=self.layers['Affine2'].dW
        grads['b2']=self.layers['Affine2'].db
        return grads

#梯度确认
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
x_batch=x_train[:3]
t_batch=t_train[:3]
grad_numerical=network.numerical_gradient(x_batch,t_batch)
grad_backprop=network.gradient(x_batch,t_batch)

#求各个权重的绝对误差的平均值
for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_numerical[key]-grad_backprop[key]))  #np.abs求绝对值
    print(key+":"+str(diff))