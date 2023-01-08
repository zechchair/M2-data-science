"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
def print(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))
############## Task 1

##################
G = nx.read_edgelist("../datasets/CA-HepTh.txt",delimiter="\t",comments="#")
G.number_of_nodes() # Number of nodes
G.number_of_edges() #Number of edges
##################


############## Task 2

##################
nx.is_directed(G)
connected_c=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
length_components=len(connected_c)
print(f"{length_components = } ")
largest_cc = G.subgraph(connected_c[0]).copy()
largest_number_nodes=largest_cc.number_of_nodes()
largest_number_edges=largest_cc.number_of_edges()
print(f"{largest_number_nodes = }")
print(f"{largest_number_edges = }")
##################


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
mean_degree_sequence = np.mean(degree_sequence)
max_degree_sequence = np.max(degree_sequence)
min_degree_sequence = np.max(degree_sequence)
median_degree_sequence = np.median(degree_sequence)
print(f"{mean_degree_sequence = }")
print(f"{max_degree_sequence = }")
print(f"{min_degree_sequence = }")
print(f"{median_degree_sequence = }")
##################
# c 2 of n + j **2 
n_c = 100
n_b = 50
n_set = 2

print(f'The number of edges in graph G consisting of two connected components\
    , where one of the connected components is a complete graph on 100 vertices\
    and the other connected component is a complete bipartite graph with 50\
    vertices in each partition set is {n_c*(n_c-1)/2 + n_b**n_set}')

print(f'The number of triangles in complete graph of 100 nodes is {n_c*(n_c-1)*(n_c-2)/6}')
print(f'The number of triangles in bipartie graphs 0')
##################



############## Task 4

##################
freq_degree = nx.degree_histogram(G) 
plt.hist(freq_degree)

plt.xscale('log')
plt.hist(freq_degree, log=True)
##################




############## Task 5

##################
G_clust_coeff = nx.transitivity(G)
print(f"{G_clust_coeff = :.2f}")

print("The maximum could be 1, When all the triplet are closed and we can find it in ")
##################
