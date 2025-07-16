# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)
    

def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad
    

# def softmax(x):
#     if x.ndim == 2:
#         x = x.T
#         x = x - np.max(x, axis=0)
#         y = np.exp(x) / np.sum(np.exp(x), axis=0)
#         return y.T 

#     x = x - np.max(x) # 溢出对策
#     return np.exp(x) / np.sum(np.exp(x))
def softmax(a):
    C = np.max(a)
    exp_a = np.exp(a-C)     # 解决溢出问题
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a


def mean_squared_error(y, t):          #均方误差
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):         
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)       #将一维数组转换为二维数组
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# def cross_entropy_error(y,t):          #单数据的交叉熵误差，y:输出,t:正确标签
#     delta=1e-7
#     return -np.sum(t*log(y+delta))

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)
