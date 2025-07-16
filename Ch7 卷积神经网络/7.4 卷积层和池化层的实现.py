import sys,os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from commonpackage.util import im2col
import numpy as np

# x1=np.random.rand(1,3,7,7)
# col1=im2col(x1,5,5,stride=1,pad=0)
# print(col1.shape)

class Convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w=w
        self.b=b
        self.stride=stride
        self.pad=pad
    def forward(self,x):
        FN,C,FH,FW=self.w.shape
        N,C,H,W=x.shape
        out_h=int(1+(H+2*self.pad-FH)/self.stride)
        out_w=int(1+(W+2*self.pad-FW)/self.stride)
        
        col=im2col(x,FH,FW,self.stride,self.pad)
        col_W=self.w.reshape(FN,-1).T#滤波器的展开，reshape自动计算-1维度上的元素个数
        out=np.dot(col,col_W)+self.b

        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad
    def forward(self,x):
        N,C,H,W=x.shape
        out_h=int(1+(H-self.pool_h)/self.stride)
        out_W=int(1+(W-self.pool_w)/self.stride)

        #展开
        col=im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col=col.reshape(-1,self.pool_h*self.pool_w)

        #最大值
        out=np.max(col,axis=1)
        #转换为合适的输出大小
        out=out.reshape(N,out_h,out_W,C).transpose(0,3,1,2)
        
        return out