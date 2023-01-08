"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)

############## Task 5
# Visualizes the karate network

##################
nx.draw(G,node_color=["blue" if label else "green" for label in y ], with_labels=True)
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
LR = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred=LR.predict(X_test) #Return the predictions
accuracy_score(y_test, y_pred)
##################


############## Task 8
# Generates spectral embeddings

##################
embedding = SpectralEmbedding(n_components=2)
X_transformed = embedding.fit_transform(np.concatenate((X_train,X_test)))
X_transformed.shape

X_train_transformed = X_transformed[idx_train,:]
X_test_transformed = X_transformed[idx_test,:]


LR_spect = LogisticRegression(random_state=0).fit(X_train_transformed, y_train)
y_pred=LR_spect.predict(X_test_transformed) #Return the predictions
accuracy_score(y_test, y_pred)
##################
