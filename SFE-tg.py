#coding:utf-8
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import math


size = 10
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
#随机生成信号序列

a = np.random.randint(0, 2, size)   #产生随机整数序列
m = np.zeros(len(t), dtype=np.float32)    #产生一个给定形状和类型的用0填充的数组
for i in range(len(t)):
    m[i] = a[int(math.floor(t[i]))]
    
ts1 = np.arange(0, (np.int64(1/sampling_t) * size))/(10*(m+1))
fsk = np.cos(np.dot(2 * pi, ts1) + pi / 4)

plt.plot(t, fsk, color='red', label='tg')
plt.legend() # 显示图例
plt.xlabel('t')
plt.ylabel('fsk')
plt.show()