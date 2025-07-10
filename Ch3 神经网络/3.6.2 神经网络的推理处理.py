import sys, os
#sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from commonpackage.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():   #读入已学习到的权重参数
    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 sample_weight.pkl 文件的绝对路径
    file_path = os.path.join(script_dir, 'sample_weight.pkl')
    with open(file_path, 'rb') as f:
        network = pickle.load(f)    
    return network                   #得到的结果为字典类型


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y   #返回的y依然是numpy数组


x_test, t_test = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
accuracy_cnt = 0
for i in range(len(x_test)):
    y = predict(network, x_test[i])
    result= np.argmax(y)      # 获取概率最高的元素的索引作为预测结果
    if result == t_test[i]:
        accuracy_cnt += 1
print("Accuracy:"+ str(accuracy_cnt/len(x_test)))
# print(x_test.shape)
# print(x_test[0].shape)
# print(W1.shape)
# print(W2.shape)
# print(W3.shape)

# #批处理
batch_size=100      #批数量
accuracy=0
for i in range(0,len(x_test),batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = predict(network,x_batch)
    result = np.argmax(y_batch,axis=1)
    accuracy += np.sum(result==t_test[i:i+batch_size])
print("batch accuracy:",accuracy/len(x_test))