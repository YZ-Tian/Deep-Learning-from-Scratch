# import numpy as np
# import matplotlib.pyplot as plt
# #生成数据
# x=np.arange(0,6,0.1)
# y1=np.sin(x)
# y2=np.cos(x)

# #绘制图形
# plt.plot(x,y1,label='sinx')
# plt.plot(x,y2,linestyle='--',label='cosx')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("sin & cos")
# plt.legend(loc="best")   #添加图例，前提是要在plt.plot()中设置一个label
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.image import imread
img=imread('./dataset/suda.jpg')
plt.imshow(img)
plt.show()