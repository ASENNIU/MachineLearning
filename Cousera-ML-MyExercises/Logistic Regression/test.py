import numpy as np

a = np.mat([1,2,3])
b = 1 / (1 + np.exp(-a))
print(b)