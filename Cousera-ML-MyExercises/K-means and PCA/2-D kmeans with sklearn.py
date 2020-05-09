import pandas as pd
from sklearn.cluster import KMeans
import scipy.io as sio
from functions import combine_data_C
import seaborn as sns
import matplotlib.pyplot as plt

mat = sio.loadmat('data/ex7data2.mat')
data = pd.DataFrame(mat.get('X'), columns=['X1', 'X2'])

sk_kmeans = KMeans(n_clusters=3)
sk_kmeans.fit(data)

sk_C = sk_kmeans.predict(data)
data_with_c = combine_data_C(data, sk_C)

sns.lmplot('X1', 'X2', hue='C', data=data_with_c, fit_reg=False)
plt.show()
