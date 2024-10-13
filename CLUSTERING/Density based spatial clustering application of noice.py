import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate sample data
X, _ = make_moons(n_samples=300, noise=0.05)

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=0.2, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

# Plot the clusters
plt.scatter(X[y_dbscan == 0, 0], X[y_dbscan == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_dbscan == 1, 0], X[y_dbscan == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[y_dbscan == -1, 0], X[y_dbscan == -1, 1], s=100, c='black', label='Noise')
plt.title('DBSCAN Clustering')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()
