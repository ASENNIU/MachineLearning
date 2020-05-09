import scipy.io as sio
from f_PCA import *
from sklearn.decomposition import PCA

mat = sio.loadmat('data/ex7faces.mat')
X = mat.get('X')

"""
plot_n_image(X, n=64)

U, S, V = PCA(X)

plot_n_image(U, n=36)

Z = project_data(X, U, k=100)
plot_n_image(Z, n=64)

X_recover = recover_data(Z, U)
plot_n_image(X_recover, n=64)
"""

sk_pca = PCA(n_components=100)
Z = sk_pca.fit_transform(X)
print(Z.shape)

plot_n_image(Z, n=64)

X_recover = sk_pca.inverse_transform(Z)
plot_n_image(X_recover, n=64)