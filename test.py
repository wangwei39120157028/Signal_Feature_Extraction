import numpy as np
import math
size = 10
sampling_t = 0.01

t = np.arange(0, size, sampling_t)
m = np.zeros(len(t), dtype=np.float32)    
a = np.random.randint(0, 2, size)
for i in range(len(t)):
    m[i] = a[int(math.floor(t[i]))]
print m