#coding:utf-8
import numpy as np
from math import pi
import matplotlib.pyplot as plt
import math


size = 10
sampling_t = 0.01
t = np.arange(0, size, sampling_t)
fsk = np.cos(np.dot(2 * pi, t) + pi / 4)
plt.plot(t, fsk, color='red', label='t')
plt.legend() # 显示图例
plt.xlabel('t')
plt.ylabel('fsk')
plt.show()