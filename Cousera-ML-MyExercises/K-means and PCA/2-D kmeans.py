import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from functions import *
import seaborn as sns

mat = sio.loadmat('data/ex7data2.mat')
data2 = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])

plt.scatter(data2['X1'], data2['X2'])
plt.show()

C, centroids, cost = k_means(data2, 3)
data_with_c = combine_data_C(data2, C)

sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
plt.show()

