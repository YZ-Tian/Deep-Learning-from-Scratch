import numpy as np
import matplotlib.pyplot as plt
def numerical_diff(f,x):    #数值微分
    h=1e-4
    return (f(x+h)-f(x))/h

# def func_1(x):
#     return 0.01*x**2 + 0.1*x

# x = np.arange(0,20,0.1)
# y = func_1(x)
# plt.plot(x,y)
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.show()

def func_2(x):
    return np.sum(x**2)

def numerical_gradient(f,x):     #求解梯度（由全部变量的偏导数汇总成的向量）
    h=1e-4
    grad=np.zeros_like(x)       #生成和x形状一样，所有元素都为0的数组
    for i in range(x.size):
        temp_x=x[i]
        x[i]=temp_x+h           #这里直接修改了数组x的值
        fxh1=f(x)
        x[i]=temp_x-h
        fxh2=f(x)
        x[i]=temp_x             #计算完后恢复x[i]的原始值
        grad[i]=(fxh1-fxh2)/(2*h)

    return grad
print(numerical_gradient(func_2,np.array([3.0,4.0])))


def gradient_descent(f,init_x,lr=0.01,step_num=100):      #梯度下降法
    x=init_x                    #设定初始值
    for i in range(step_num):
        grad=numerical_diff(f,x)
        x-=lr*grad
    return x

