import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("D:/Machine Learning/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv")
x = df.iloc[:,[3,4]].values

#using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.title ('Dendogam')
plt.xlabel('Customers')
plt.ylabel("Euclidean Distance") 
plt.show()
   
#training hierarchical model on dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, linkage = 'ward')
y_hc = hc.fit_predict(x)
#visualising the clusters 

plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
