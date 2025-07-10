#回归问题用恒等函数，分类问题用softmax函数（表示类别概率）
import numpy as np
# def softmax(a):
#     exp_a=np.exp(a)
#     sum_exp_a=np.sum(exp_a)
#     return exp_a/sum_exp_a
def softmax(a):
    C=np.max(a)
    exp_a=np.exp(a-C)     # 解决溢出问题
    sum_exp_a=np.sum(exp_a)
    return exp_a/sum_exp_a

a=np.array([0.3,2.9,4.0])
print(softmax(a))