import sys,os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
print(x_train.shape)    #(6000,784)

# mini-batch采样
train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)    #从0~train_size中随机选batch_size个数字，生成一个数组
x_batch=x_train[batch_mask]
t_batch=t_train[batch_mask]


# mini-batch版本的交叉熵误差
def cross_entropy_error(y, t):         
     #如果输入为单样本，将一维数组转换为二维数组
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)      
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    # 1. 提取每个样本真实标签对应的预测概率
    batch_size = y.shape[0]
    correct_probs = y[np.arange(batch_size), t]  # 形状：(batch_size,)

    # 2. 添加微小值防止对数计算时出现数值不稳定（如 log(0)）
    safe_probs = correct_probs + 1e-7

    # 3. 对概率取自然对数
    log_probs = np.log(safe_probs)  # 形状：(batch_size,)

    # 4. 计算所有样本的负对数损失之和
    sum_loss = -np.sum(log_probs)

    # 5. 计算平均损失
    avg_loss = sum_loss / batch_size

    return avg_loss         
# batch_size = y.shape[0]    #表示样本数量
# return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
