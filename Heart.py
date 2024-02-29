

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn import metrics
import matplotlib.pyplot as plt
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors

x = pd.read_csv("D:/Heart-disease-prediction/heart_statlog_cleveland_hungary_final.csv")
heart_data = x.drop(columns='target',axis=1)
x1=heart_data[["age","cholesterol"]]
x1.isnull().sum()
x1.describe()

plt.scatter(x1.cholesterol,x1.age)
plt.xlabel("cholesterol")
plt.ylabel("age")

#DBSCAN clustering
db = DBSCAN(eps=40,min_samples=5)
db.fit(x1)
x_pred = db.fit_predict(x1)

core_samples_mask = np.zeros_like(db.labels_,dtype=bool)

core_samples_mask[db.core_sample_indices_] = True


n_clusters_ = len(set(x_pred)) - (1 if -1 in x_pred else 0)
n_noise_ = list(x_pred).count(-1)

colours={}
colours[-1]='b'
colours[0]='r'
colours[1]='g'
colours[2]='y'
colours[3]='c'
colours[4]='m'
cvec=[colours[label] for label in x_pred]

k=plt.scatter(x1['cholesterol'],x1['age'],color='b')
r=plt.scatter(x1['cholesterol'],x1['age'],color='r')
g=plt.scatter(x1['cholesterol'],x1['age'],color='g')
y=plt.scatter(x1['cholesterol'],x1['age'],color='y')
c=plt.scatter(x1['cholesterol'],x1['age'],color='c')
m=plt.scatter(x1['cholesterol'],x1['age'],color='m')
plt.figure(figsize=(5,5))
plt.scatter(x1['cholesterol'],x1['age'],color=cvec)

plt.legend((k,r,g,y,c,m),('Outliers','0th Cluster','1st Cluster','2nd cluster','3rd cluster','4th cluster'))

plt.show()

silhouette_score(x1,x_pred)*100

