import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd

mat = sio.loadmat('data/ex7data1.mat')
#print(mat.keys())

datal = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])
#print(datal.head(5))

mat2 = sio.loadmat('data/ex7data2.mat')
data2 = pd.DataFrame(mat2.get('X'), columns=['X1', 'X2'])

fig, ax = plt.subplots(1,2, figsize=(18,6))
ax[0].scatter(datal['X1'], datal['X2'])
ax[1].scatter(data2['X1'], data2['X2'])
plt.show()
