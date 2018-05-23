# NADJA HERGER, 2018 - nadja.herger@student.unsw.edu.au

#####################################################
## Import necessary modules
#####################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import euclidean_distances
from mpl_toolkits.mplot3d import Axes3D
import scipy
from sklearn import manifold


#####################################################
## Generate random data
#####################################################
ndots = 20
seed = np.random.RandomState(seed=3)
x = np.array([seed.randint(0, 20, ndots).astype(np.float) for i in range(3)]).T  # (20, 3)


#####################################################
## 3D scatter plot
#####################################################
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


#####################################################
## Calculate dissimilarity matrix
#####################################################
D = euclidean_distances(x)  # (20, 20)
D_ssd = scipy.spatial.distance.squareform(D)  # (4005,)


#####################################################
## Hierarchical clustering
#####################################################
Z = linkage(D_ssd)
plt.figure(figsize=(16, 8))
plt.ylabel('Dissimilarity', fontsize=14)
dend = dendrogram(Z, leaf_font_size=14, distance_sort='ascending')
ax = plt.gca()
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.show()


#####################################################
## Multidimensional Scaling
#####################################################
mds = manifold.MDS(n_components=2, dissimilarity='precomputed', random_state=3)
Y = mds.fit_transform(D)  # (20, 2)
mds.stress_


#####################################################
## 2D scatter plot
#####################################################
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(Y[:, 0], Y[:, 1])
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.show()


#####################################################
## Shepard Diagram
#####################################################
D_repr = scipy.spatial.distance_matrix(Y, Y)  # (20, 20)
fig = plt.figure()
plt.plot(D, D_repr, marker='o', markersize=4, linestyle='None')
x = np.linspace(0, np.max(np.concatenate([D, D_repr])), 30)
plt.plot(x, x, linestyle=':', color='k')
plt.xlabel('Original distances', fontsize=14)
plt.ylabel('MDS distances', fontsize=14)
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.show()
