import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_X(df):
    """
    use concat to add intersect feature to avoid side effect
    not efficient for big dataset though
    """

    ones = pd.DataFrame({'ones':np.ones(len(df))})
    data = pd.concat([ones, df], axis=1)
    return data.iloc[:, :-1].to_numpy()

def get_y(df):
    return np.array[df.iloc[:, :-1]]

def normlize_feature(df):
    df.apply(lambda column: (column - column.mean()) / column.std())

def plot_n_image(X, n):
    pic_size = int(np.sqrt(X.shape[1]))
    grid_size = int(np.sqrt(n))

    first_n_image = X[:n, :]

    fig, ax_array = plt.subplots(nrows=grid_size, ncols=grid_size, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(grid_size):
        for c in  range(grid_size):
            ax_array[r, c].imshow(first_n_image[grid_size * r + c].reshape((pic_size, pic_size)))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()

def covariance_matrix(X):
    """
    :param X: X(ndarray) (m, n)
    :return: cov_mat(ndarray) (n, n) covariance matrix of X
    """
    return (X.T @ X) / X.shape[0]

def normlize(X):
    X_copy = X.copy()
    return (X_copy - X_copy.mean(axis=0)) / X_copy.std(axis=0)

def PCA(X):
    X_norm = normlize(X)

    Sigma = covariance_matrix(X_norm)

    U, S, V = np.linalg.svd(Sigma)

    return U, S, V

def project_data(X, U, k):
    #projected X (n dim) at k dim

    m, n = X.shape

    if k > n:
        raise ValueError('k should be lower dimension of n')

    return X @ U[:, :k]

def recover_data(Z, U):
    m, n = Z.shape

    if n >= U.shape[0]:
        raise ValueError('Z dimension is >= U, you should recover from lower dimension to higher')

    return Z @ U[:, :n].T


