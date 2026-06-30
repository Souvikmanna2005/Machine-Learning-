#K-Means

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#s1:same dataset create
X,y=make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.60,
    random_state=42
    )
#s2:apply k-means clustering
kmeans=KMeans(
    n_clusters=4,
    random_state=42
    )
kmeans.fit(X)

#predicted cluster labels
labels=kmeans.predict(X)

#cluster centers
centroids=kmeans.cluster_centers_
print(centroids)

#s3:visualize clusters
plt.figure(figsize=(8,6))

#plot data points
plt.scatter(
    X[:,0],
    X[:,1],
    c=labels,
    cmap='viridis',
    s=50
    )

#plot centroids
plt.scatter(
    centroids[:,0],
    centroids[:,1],
    c='red',
    marker='X',
    s=200,
    label='Centroids'
    )
plt.show()