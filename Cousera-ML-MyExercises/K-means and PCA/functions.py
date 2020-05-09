import numpy as np
import pandas as pd

def combine_data_C(data, C):
    data_with_c = data.copy()
    data_with_c['C'] = C
    return data_with_c

def random_init(data, k):
    """

    :param data: DataFrame
    :param k: int
    :return: k simples ndarrays
    """

    return data.sample(k).to_numpy()

def _find_your_cluster(x, centroids):
    distance = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=centroids-x)
    #将数组沿轴转递给函数，arr，axis，func1d
    #linagl线性代数模块，norm求范数，默认l2范数
    return np.argmin(distance)

def assign_cluster(data, centoids):
    return np.apply_along_axis(lambda x: _find_your_cluster(x, centoids), axis=1, arr=data.to_numpy())

def new_centroids(data, C):
    data_with_c = combine_data_C(data, C)
    return data_with_c.groupby('C', as_index=False).mean().sort_values(by='C').drop('C', axis=1).to_numpy()

def cost(data, centorids, C):
    m = data.shape[0]
    expand_C_with_centroids = centorids[C]
    distance = np.apply_along_axis(func1d=np.linalg.norm, axis=1, arr=data.to_numpy()- expand_C_with_centroids)
    return distance.sum() / m

def _k_means_iter(data, k, epoch=100, tol=0.0001):
    centroids = random_init(data, k)
    cost_process = []

    for i in range(epoch):
        C = assign_cluster(data, centroids)
        centroids = new_centroids(data, C)
        cost_process.append(cost(data, centroids, C))
        print('Running epoch{}: {}'.format(i, cost_process[-1]))

        if len(cost_process) > 1:
            if(np.abs(cost_process[-1] - cost_process[-2])) / cost_process[-1] < tol:
                break

    return C, centroids, cost_process[-1]

def k_means(data, k, epoch=100, n_init=10):
    """
    do multiple random init and pick the best one to return
    :param data: DataFrame
    :return: C, centriods, least_cost
    """

    tries = np.array([_k_means_iter(data, k, epoch) for _ in range(n_init)])

    least_cost_idx = np.argmin(tries[:,-1])

    return tries[least_cost_idx]

