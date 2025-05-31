# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 12:53:12 2020

@author: AymenHAMROUNI
"""
import pandas as pd
pd.set_option('display.max_colwidth', 500)
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

DatasetPath= '../../facebookDataset'
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
Featurefile='../../facebookDataset/Facebook.emb'
file1 = open(Featurefile, 'r') 
Lines = file1.readlines() 
count=0
Dict={}
for line in Lines:
    if  2 < len(line.split(" ")):
        nodeId=line.split(" ")[0]
        Dict[int(nodeId)]=line.split(" ")[1:]
Graphembedding=pd.DataFrame(Dict).T

from scipy.spatial.distance import cdist 


with open("../../embedding"+'.pkl', 'rb') as f:
        embeddings=pickle.load(f)
        
        
Graphembedding=pd.DataFrame(embeddings).T

Graphembedding=Graphembedding.drop(['id'])

Graphembedding=pd.DataFrame(Graphembedding).T

# Instantiating PCA
pca = PCA()

# Fitting and Transforming the DF
df_pca = pca.fit_transform(Graphembedding)

# Plotting to determine how many features should the dataset be reduced to
# plt.style.use("bmh")
# plt.figure(figsize=(14,4))
# plt.plot(range(1,Graphembedding.shape[1]+1), pca.explained_variance_ratio_.cumsum())
# plt.show()

# Finding the exact number of features that explain at least 95% of the variance in the dataset
total_explained_variance = pca.explained_variance_ratio_.cumsum()
n_over_95 = len(total_explained_variance[total_explained_variance>=.95])
n_to_reach_95 = Graphembedding.shape[1] - n_over_95

# Printing out the number of features needed to retain 95% variance
print(f"Number features: {n_to_reach_95}\nTotal Variance Explained: {total_explained_variance[n_to_reach_95]}")

# Reducing the dataset to the number of features determined before
pca = PCA(n_components=n_to_reach_95)

# Fitting and transforming the dataset to the stated number of features and creating a new DF
df_pca = pca.fit_transform(Graphembedding)

# Seeing the variance ratio that still remains after the dataset has been reduced
print(pca.explained_variance_ratio_.cumsum()[-1])








# Setting the amount of clusters to test out
cluster_cnt = [i for i in range(12,13, 1)]

# Establishing empty lists to store the scores for the evaluation metrics
s_scores = []

db_scores = []
distortions=[]
inertias=[]
mapping1={}
mapping2={}
# Looping through different iterations for the number of clusters
for i in cluster_cnt:
    
    # Hierarchical Agglomerative Clustering with different number of clusters
    # hac = AgglomerativeClustering(n_clusters=i)
    
    # hac.fit(df_pca)
    
    # cluster_assignments = hac.labels_
    
    # ## KMeans Clustering with different number of clusters
    k_means = KMeans(n_clusters=i)
    
    k_means.fit(df_pca)
    
    cluster_assignments = k_means.predict(df_pca) 
      
    distortions.append(sum(np.min(cdist(df_pca, k_means.cluster_centers_, 
                      'euclidean'),axis=1)) / df_pca.shape[0]) 
    inertias.append(k_means.inertia_) 
  
    mapping1[i] = sum(np.min(cdist(df_pca, k_means.cluster_centers_, 
                  'euclidean'),axis=1)) / df_pca.shape[0] 
    mapping2[i] = k_means.inertia_ 
    
    # Appending the scores to the empty lists    
    s_scores.append(silhouette_score(df_pca, cluster_assignments))
    
    db_scores.append(davies_bouldin_score(df_pca, cluster_assignments))
    
def plot_evaluation(y, x=cluster_cnt):
    """
    Plots the scores of a set evaluation metric. Prints out the max and min values of the evaluation scores.
    """
    
    # Creating a DataFrame for returning the max and min scores for each cluster
    df = pd.DataFrame(columns=['Cluster Score'], index=[i for i in range(2, len(y)+2)])
    df['Cluster Score'] = y
    
    print('Max Value:\nCluster #', df[df['Cluster Score']==df['Cluster Score'].max()])
    print('\nMin Value:\nCluster #', df[df['Cluster Score']==df['Cluster Score'].min()])
    
    # Plotting out the scores based on cluster count
    plt.figure(figsize=(16,6))
    plt.style.use('ggplot')
    plt.plot(x,y)
    plt.xlabel('# of Clusters')
    plt.ylabel('Score')
    plt.show()
    
# Running the function on the list of scores
plot_evaluation(s_scores)

plot_evaluation(db_scores)
plt.figure()

plt.plot(cluster_cnt, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 
plt.figure()

plt.plot(cluster_cnt, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show() 


plt.figure()
plt.scatter(df_pca[:,0],df_pca[:,1], c=cluster_assignments, cmap='rainbow')
plt.scatter(k_means.cluster_centers_[:,0] ,k_means.cluster_centers_[:,1], color='black')














