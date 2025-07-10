import sys,os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
#sys.path.append(os.pardir)       #为了导入父目录中的文件而进行的设定
#sys.path用于返回导入模块时搜索的目录列表，os.path表示父目录
from dataset.mnist import load_mnist
from PIL import Image        #图像显示用PIL模块

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))   #把数组的图像数据转换为PIL用的数据对象
    pil_img.show()

(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
img=x_train[1]
label=t_train[1]
print(label)     #5

print(img.shape)     #(784,)flatten之后变成784个元素的一维数组了
img=img.reshape((28,28))    #显示图象时将其还原为原来的28*28像素的形状
print(img.shape)    #(28,28)

img_show(img)