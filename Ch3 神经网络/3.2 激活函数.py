import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
#阶跃函数
def step_function(x):
    # y=x>0
    # y1=y.astype(np.int32)
    # return y1
    return np.array(x>0,dtype=int)
x=np.arange(-5,5,0.1)
y1=step_function(x)
plt.plot(x,y1,label="阶跃函数")

#sigmoid函数
def sigmoid(x):
    return 1/ ( 1 + np.exp(-x) )  #由于Numpy的广播功能，返回的是数组
y2=sigmoid(x)
plt.plot(x,y2,label="sigmoid函数")
plt.legend()
plt.show()

#ReLU(Recitified Linear Unit)函数
def relu(x):
    return np.maximum(0,x)
y3=relu(x)
plt.plot(x,y3,label="ReLU函数")
plt.legend()
plt.show()
