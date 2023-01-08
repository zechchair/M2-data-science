"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans

np.set_printoptions(precision=2)
def print(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))
############## Task 1

##################
G = nx.read_edgelist("../datasets/CA-HepTh.txt",delimiter="\t",comments="#")

############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    A = nx.adjacency_matrix(G)
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    
    Lrw = eye(G.number_of_nodes()) - D_inv @ A
    
    eigen_values, eigen_vectors = eigs(Lrw, k=k, which='SR')
    
    eigen_vectors = np.real(eigen_vectors)
    kmeans = KMeans(n_clusters=k).fit(eigen_vectors)
    
    clustering = {node: kmeans.labels_[i] for i, node in enumerate(G.nodes())}
    ##################
    

    
    return clustering




############## Task 7

##################
G = nx.read_edgelist("../datasets/CA-HepTh.txt",delimiter="\t",comments="#")
connected_comp = nx.connected_components(G)
largest_cc = max(connected_comp, key=len)
giant_cc = G.subgraph(largest_cc).copy()
k=50
dict_cluster = spectral_clustering(giant_cc, k)
print('\n\n',dict_cluster)
##################






############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    m = G.number_of_edges()
    clusters = set(clustering.values()) # unique clusters  
    modularity = 0
    for c in clusters:
        nodes = [node for node in G.nodes() if clustering[node] == c]
        subgraph = G.subgraph(nodes)
        lc = subgraph.number_of_edges()
        dc = sum(G.degree(node) for node in nodes)
        modularity += lc/m - (dc/(2*m))**2
    ##################
    
    return modularity



############## Task 9

##################
# your code here #
print("Modularity of the giant component: ",modularity(giant_cc, dict_cluster))
random_clustering = {node: randint(0, k-1) for node in giant_cc.nodes()}
print("Modularity of a random component: ",modularity(giant_cc, random_clustering))
##################





