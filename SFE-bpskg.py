#coding:utf-8
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import math



size = 10
sampling_t = 0.1
t = np.arange(0, size, sampling_t)
bpsk = []
a = np.random.randint(0, 2, 100) - 1  #产生随机整数序列
m = np.zeros(len(t), dtype=np.float32)    #产生一个给定形状和类型的用0填充的数组
for i in range(len(t)):
    m[i] = a[int(math.floor(t[i]))]
for aa in a:
    fsk = aa * np.cos(np.dot(2 * pi, t) + pi / 4)
    bpsk.append(fsk)
plt.plot(t, bpsk, color='red', label='t')
plt.legend() # 显示图例
plt.xlabel('t')
plt.ylabel('fsk')
plt.show()














