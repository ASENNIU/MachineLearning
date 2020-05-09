import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from f_PCA import *

mat = sio.loadmat('data/ex7data1.mat')

data = mat.get('X')

X_norm = normlize(data)

U, S, V = PCA(X_norm)

Z = project_data(X_norm, U, 1)
X_recover = recover_data(Z, U)

#project data to lower dimension

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.regplot('X1', 'X2', data=pd.DataFrame(data, columns=['X1', 'X2']), fit_reg=False, ax=ax[0])
ax[0].set_title('Original dimension')

sns.rugplot(Z, ax=ax[1])
ax[1].set_xlabel('Z')
ax[1].set_title('Z dimension')
plt.show()

#recover data to original dimension

_fig, _ax = plt.subplots(1, 3, figsize=(12, 4))

sns.rugplot(Z, ax=_ax[0])
_ax[0].set_title('Z dimension')
_ax[0].set_xlabel('Z')

sns.regplot('X1', 'X2', data=pd.DataFrame(X_recover, columns=['X1', 'X2']), fit_reg=False, ax=_ax[1])
_ax[1].set_title("2D projection from Z")

sns.regplot('X1', 'X2', data=pd.DataFrame(X_norm, columns=['X1', 'X2']), fit_reg=False, ax=_ax[2])
_ax[2].set_title('Original dimension')
plt.show()

