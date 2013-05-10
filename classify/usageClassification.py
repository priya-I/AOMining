'''
Created on Apr 19, 2013

@author: priya
'''
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import Scaler
from sklearn.datasets.samples_generator import make_blobs

import numpy as np



km=KMeans(n_clusters=10)

myMatrix=[];
inputfile="cluster_sample.csv"
users=[]
gd=3651201
gs=131070
with open(inputfile,"r") as inputf:
    for row in inputf:
       # print row
        record=row.split(",")
        duration=record[0]
        session=record[1]
        userid=record[2].strip("\n")
        users.append(userid)
        myMatrix.append([int(duration),int(session)])
     
#print myMatrix   
inputf.close();    
km.fit(myMatrix);
labels=km.labels_
centers=km.cluster_centers_
km.fit_predict(myMatrix);
km.fit_transform(myMatrix)
print myMatrix
# Generate sample data
'''
centers = centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=len(myMatrix), centers=centers, cluster_std=0.4,
                            random_state=0)

X = Scaler().fit_transform(X)

##############################################################################
# Compute DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
core_samples = db.core_sample_indices_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
      

centers = [[20000,3000], [6000,120], [120000, 5000]]
X, labels_true = make_blobs(n_samples=len(myMatrix), centers=5, cluster_std=0.1)
#print X
#print labels_true
myMatrix = Scaler().fit_transform(X)
db=DBSCAN().fit(myMatrix)
db.fit_predict(myMatrix)
core_samples = db.core_sample_indices_
labels = db.labels_

for label in labels:
    print label 
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
'''


