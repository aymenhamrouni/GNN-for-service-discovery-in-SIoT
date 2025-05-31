import community
import networkx as nx
from collections import Counter
from itertools import chain
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community.quality import coverage
from networkx.algorithms.community.quality import performance
from node2vec import Node2Vec
import matplotlib.pyplot as plt
import numpy as np

import _pickle as pickle

import argparse
import numpy as np
import networkx as nx
#import node2vec
from gensim.models import Word2Vec
from scipy.spatial.distance import cdist 

import statistics

import time

import pandas as pd



import pandas as pd
pd.set_option('display.max_colwidth', 500)
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict




EMBEDDING_FILENAME = './embeddings'
EMBEDDING_MODEL_FILENAME = './embeddings_model'


def plot_evaluation(y, x):
    """
    Plots the scores of a set evaluation metric. Prints out the max and min values of the evaluation scores.
    """
    
    # Creating a DataFrame for returning the max and min scores for each cluster
    df = pd.DataFrame(columns=['Cluster Score'], index=[i for i in range(2, len(y)+2)])
    df['Cluster Score'] = y
    
    #print('Max Value:\nCluster #', df[df['Cluster Score']==df['Cluster Score'].max()])
    #print('\nMin Value:\nCluster #', df[df['Cluster Score']==df['Cluster Score'].min()])
    
    # Plotting out the scores based on cluster count
    plt.figure(figsize=(16,6))
    plt.style.use('ggplot')
    plt.plot(x,y)
    plt.xlabel('# of Clusters')
    plt.ylabel('Score')
    plt.show()
    
    
# def learn_embeddings(walks,dimensions=128,output='./embeddings.emb',window_size=10,workers=8,iteration=5):
#     '''
#     Learn embeddings by optimizing the Skipgram objective using SGD.
#     '''
#     walks = [map(str, walk) for walk in walks]
#     model = Word2Vec(walks, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iteration)
#     model.wv.save_word2vec_format(output)

def commmunityDetection_basic(graphComm):
    # Read the adjencey matrix and convert it to dataframe
    #df = pd.DataFrame(edgeList)

    #print(df)
    #G_Comm = nx.read_edgelist(edgeList)
    #G_Comm = nx.from_pandas_adjacency(df)
    G_Comm = graphComm

    # first compute the best partition using Lovuain Algorithm
    # community.best_partition(graph, partition=None, weight='weight', resolution=1.0, randomize=None, random_state=None)
    start_time = time.time()
    
    partition = community.best_partition(G_Comm, weight='weight', random_state=10)

    # Count the time of processing and finding communities
    total_time = time.time() - start_time

    # Measures
    # number of communities
    numberOfCommunities = len(set(partition.values()))

    #print("Number of Communities", numberOfCommunities)

    # Largest community size
    maxVal = 0
    communityLabelsOnly = list(partition.values())

    res = communityLabelsOnly[0] 
    for i in communityLabelsOnly: 
        freq = communityLabelsOnly.count(i) 
        if freq > maxVal: 
            maxVal = freq 
            res = i 
    largestCommunitySize = communityLabelsOnly.count(res)

    # Avrage Size of communities
    d = {x:communityLabelsOnly.count(x) for x in communityLabelsOnly}
    freq = list (d.values())
    avrageSizeCommunities = statistics.mean(freq)
    #print(partition)
    # Begin ------------- to calcaute the COVREAGE and PERFORMANCE
    # Calculate coverage: The coverage of a partition is the ratio of the number of intra-community edges 
    # to the total number of edges in the graph.
    # Make sure to convert the partition list to something look like that = [{0, 1}, {2, 3, 4}, {5}]
    QualityPartition = []
    for communityLabelsNew in range(0,numberOfCommunities):
        listOfNodes = [key  for (key, value) in partition.items() if value == communityLabelsNew]
        QualityPartition.append(set(listOfNodes))
    #print(QualityPartition)
    #print("-"*50)
    #print("Nodes in each communtiy")
    #print(QualityPartition)

    modularityScore = modularity(G_Comm, QualityPartition)
    coverageScore = coverage(G_Comm, QualityPartition)
    performanceScore = performance(G_Comm, QualityPartition)
    densityScore = nx.density(G_Comm)

    #print("Modualrity", modularityScore)
    #print("Covrage", coverageScore)

    jj = 0
    hist_values=[]
    bins=[]
    #for ii in QualityPartition:
    #    
    #    print("Community", jj, "No. Of devices", len(ii))
    #    hist_values.append(len(ii))
    #    bins.append(jj)
    #    jj = jj + 1

    #plt.hist(hist_values, bins=bins)
    #plt.show()

    print("Sum of edges weights", G_Comm.size(weight='weight'))
    # END ------------- to calcaute the COVREAGE and PERFORMANCE

    # Number of nodes
    noNodes = len(G_Comm)

    # Number of nodes
    noEdges = G_Comm.number_of_edges()
    

    return largestCommunitySize, numberOfCommunities, total_time, avrageSizeCommunities, coverageScore, noNodes, noEdges, coverageScore, performanceScore, modularityScore, densityScore


def commmunityDetection_GNN(graphComm, parameter,k,p=1,q=1,num_walks=10,walk_length=500):
   
    # Count the time of processing and finding communities
    start_time = time.time()

    # edgelist=nx.to_pandas_adjacency(graphComm)
    
    # #Embedding Phase using parameter p to determine which type of embedding
    # #to choose
    # for edge_1 in edgelist.keys():
    #     for edge_2 in edgelist.keys():
    #         if edge_1!=edge_2 and edgelist[edge_1][edge_2]==1:
    #                 with open('./edgelist.txt', 'a') as f1:
    #                     f1.write(edge_1+" "+edge_2+"\n")
    
    print("Perform Embedding with p=",str(p))
    if parameter==0:
        # '''
        # Pipeline for representational learning for all nodes in a graph, edges only.
        # '''
        # G = node2vec.Graph(graphComm, False, p, q)
        # G.preprocess_transition_probs()
        # walks = G.simulate_walks(num_walks, walk_length)
        # learn_embeddings(walks)
        #Precompute probabilities and generate walks
        
        
        
        node2vec = Node2Vec(graphComm, dimensions=64, walk_length=30, num_walks=200, workers=1)
        
        ## if d_graph is big enough to fit in the memory, pass temp_folder which has enough disk space
        # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
        #node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4, temp_folder="/mnt/tmp_data")
        
        # Embed
        model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
        
        # Look for most similar nodes
        #model.wv.most_similar('2')  # Output node names are always strings
        
        # Save embeddings for later use
        model.wv.save_word2vec_format(EMBEDDING_FILENAME+"_"+str(k)+".embed")
        
        # Save model for later use
        model.save(EMBEDDING_MODEL_FILENAME+"_"+str(k)+".model")

    elif parameter==1:
        ''' GNN for graph using the edges and atttirbutes
        '''
        print("None")
    
    
    
    
    
    
    
    
    
    
    
    print("Perform Clustering with p=",str(p))

    file = open(EMBEDDING_FILENAME+"_"+str(k)+".embed", 'r') 
    Lines = file.readlines() 
    count=0
    Dict={}
    Nodeorder={}
    for line in Lines:
        if  2 < len(line.split(" ")):
            nodeId=line.split(" ")[0]
            Dict[int(nodeId)]=line.split(" ")[1:]
            Nodeorder[count]=int(nodeId)
            count=count+1
    Embedding_pandas=pd.DataFrame(Dict).T

    
    
    # Instantiating PCA
    pca = PCA()
    
    # Fitting and Transforming the DF
    df_pca = pca.fit_transform(Embedding_pandas)
    
    # Plotting to determine how many features should the dataset be reduced to
    # plt.style.use("bmh")
    # plt.figure(figsize=(14,4))
    # plt.plot(range(1,Graphembedding.shape[1]+1), pca.explained_variance_ratio_.cumsum())
    # plt.show()
    
    # Finding the exact number of features that explain at least 95% of the variance in the dataset
    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    n_over_95 = len(total_explained_variance[total_explained_variance>=.95])
    n_to_reach_95 = Embedding_pandas.shape[1] - n_over_95
    
    # Printing out the number of features needed to retain 95% variance
    #print(f"Number features: {n_to_reach_95}\nTotal Variance Explained: {total_explained_variance[n_to_reach_95]}")
    
    # Reducing the dataset to the number of features determined before
    pca = PCA(n_components=n_to_reach_95)
    
    # Fitting and transforming the dataset to the stated number of features and creating a new DF
    df_pca = pca.fit_transform(Embedding_pandas)
    
    # Seeing the variance ratio that still remains after the dataset has been reduced
    #print(pca.explained_variance_ratio_.cumsum()[-1])
    
    
    
    
    
    
    
    
    # Setting the amount of clusters to test out
    cluster_cnt = [i for i in range(2,100, 1)]
    
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
        
    plot_evaluation(s_scores,cluster_cnt)
    
    plot_evaluation(db_scores,cluster_cnt)
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
    communityLabelsOnly={}
    for i in range(0,len(k_means.labels_)):
        communityLabelsOnly[str(Nodeorder[i])]=str(k_means.labels_[i])




    def maxocc(d):
        counts = Counter(chain.from_iterable(
            [v] if isinstance(v, str) else v for v in d.values()))
        return counts.most_common(1)[0]
    # Measures
    # number of communities
    numberOfCommunities = len(k_means.labels_)

    #print("Number of Communities", numberOfCommunities)

    # Largest community size

    largestCommunitySize = int(maxocc(communityLabelsOnly)[1])

    # Average Size of communities
    d = {x:list(communityLabelsOnly.values()).count(x) for x in communityLabelsOnly}
    freq = list (d.values())
    avrageSizeCommunities = statistics.mean(freq)

    # Begin ------------- to calcaute the COVREAGE and PERFORMANCE
    # Calculate coverage: The coverage of a partition is the ratio of the number of intra-community edges 
    # to the total number of edges in the graph.
    # Make sure to convert the partition list to something look like that = [{0, 1}, {2, 3, 4}, {5}]
    #print(communityLabelsOnly)
    QualityPartition = []
    for communityLabelsNew in range(0,numberOfCommunities):
        listOfNodes = [str(key) for (key, value) in communityLabelsOnly.items() if value == str(communityLabelsNew)]
        QualityPartition.append(set(listOfNodes))
    #print("-"*50)
    #print("Nodes in each communtiy")
    #print(QualityPartition)
    G_Comm = graphComm
    modularityScore = modularity(G_Comm, QualityPartition)
    coverageScore = coverage(G_Comm, QualityPartition)
    performanceScore = performance(G_Comm, QualityPartition)
    densityScore = nx.density(G_Comm)

    #print("Modualrity", modularityScore)
    #print("Covrage", coverageScore)

    jj = 0
    hist_values=[]
    bins=[]
    #for ii in QualityPartition:
    #    
    #    print("Community", jj, "No. Of devices", len(ii))
    #    hist_values.append(len(ii))
    #    bins.append(jj)
    #    jj = jj + 1

    #plt.hist(hist_values, bins=bins)
    #plt.show()

    print("Sum of edges weights", G_Comm.size(weight='weight'))
    # END ------------- to calcaute the COVREAGE and PERFORMANCE

    # Number of nodes
    noNodes = len(G_Comm)

    # Number of nodes
    noEdges = G_Comm.number_of_edges()
    
    total_time = time.time() - start_time

    return largestCommunitySize, numberOfCommunities, total_time, avrageSizeCommunities, coverageScore, noNodes, noEdges, coverageScore, performanceScore, modularityScore, densityScore
