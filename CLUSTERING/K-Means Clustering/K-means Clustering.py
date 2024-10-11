import numpy as np
import pandas as pd
import matplotlib.pyplot as mtp
import seaborn as sns

df = pd.read_csv("D:/Machine Learning/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv")
x = df.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters= i,init= 'k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
mtp.plot(range(1,11),wcss)
mtp.title("elbow methos")
mtp.xlabel("Number of clusters")
mtp.ylabel("wcss")
mtp.show()
    
#training the kmeans model on the dataset
kmeans = KMeans(n_clusters= 5,init='k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(x)

#visualising the clusters 
mtp.scatter(x[y_kmeans ==0,0],x[y_kmeans == 0,1],s= 100, c ='red',label = 'cluster1')
mtp.scatter(x[y_kmeans ==1,0],x[y_kmeans == 1,1], s=100, c='green',label ='cluster 2')
mtp.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1], s=100,c= "yellow", label ='cluster3')
mtp.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1], s =100, c ='pink', label = 'cluster4')
mtp.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1], s = 100, c = 'magenta', label = 'cluster 5')
mtp.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s =100 , c ='blue', label = 'centroids')
mtp.title("Clusters of customers")
mtp.xlabel("annual income")
mtp.ylabel('spending score')
mtp.legend()
mtp.show()