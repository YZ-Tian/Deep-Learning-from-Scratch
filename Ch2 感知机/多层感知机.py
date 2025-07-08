import numpy as np
#单层感知机
#与门
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    temp=np.sum(w*x)+b
    if temp<=0:
        return 0
    else:
        return 1
#与非门
def NAND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([-0.5,-0.5])
    b=0.7
    temp=np.sum(w*x)+b
    if temp<=0:
        return 0
    else:
        return 1
#或门
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    temp=np.sum(w*x)+b
    if temp<=0:
        return 0
    else:
        return 1
    
#多层感知机（异或门）
def XOR(x1,x2):
    s1=NAND(x1,x2)
    s2=OR(x1,x2)
    y=AND(s1,s2)
    return y

print(XOR(0,0))
print(XOR(1,0))
print(XOR(0,1))
print(XOR(1,1))