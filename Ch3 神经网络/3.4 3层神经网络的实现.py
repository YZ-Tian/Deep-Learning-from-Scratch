import numpy as np
def init_network():      #封装初始权重和偏置
    network={}
    network['w1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['w2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['w3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['b2']=np.array([0.1,0.2])
    network['b3']=np.array([0.1,0.2])
    return network

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

def forward(network,x):   #封装将输入信号转化为输出信号的过程
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    y=identity_function(a3)
    return y

network=init_network()
x=np.array([1.0,1.5])
print(forward(network,x))
