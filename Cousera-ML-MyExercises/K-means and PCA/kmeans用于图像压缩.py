import matplotlib.pyplot as plt
from skimage import io
from functions import *
from sklearn.cluster import KMeans

pic = io.imread('data/bird_small.png') / 255

#print(pic.shape)

data = pic.reshape(128*128, 3)

"""
C, centroids, cost = k_means(pd.DataFrame(data), 16, epoch=10, n_init=3)

compressed_pic = centroids[C].reshape((128, 128, 3))
"""

model = KMeans(n_clusters=16, n_init=10, n_jobs=-1)
model.fit(data)

centroids = model.cluster_centers_

C = model.predict(data)
compressed_pic = centroids[C].reshape((128, 128, 3))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()

