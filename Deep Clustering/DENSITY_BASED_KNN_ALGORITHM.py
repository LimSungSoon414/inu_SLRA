import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# dataset = np.array([[2.7810836,2.550537003],
# 	[1.465489372,2.362125076],
# 	[3.396561688,4.400293529],
# 	[1.38807019,1.850220317],
# 	[3.06407232,3.005305973],
# 	[7.627531214,2.759262235],
# 	[5.332441248,2.088626775],
# 	[6.922596716,1.77106367],
# 	[8.675418651,0.242068655],
# 	[7.673756466,3.508563011]])

dataset = np.array([[2.7,2.5],
	[4.3,2.4],
	[3.3,4.4],
	[1.4,1.9],
	[3.1,3.0],
	[5.3,2.7],
	[4.2,2.1],
	[3.2,1.8],
	[0.6,1.2],
	[6.3,3.5]])

k = 2
knn_model = NearestNeighbors(n_neighbors=k+1, radius=0.4).fit(dataset)

Index = []

for i in range(len(dataset)):
    data_point = dataset[i].reshape(1, -1)
    distances, indices = knn_model.kneighbors(data_point)
	Index.append(indices[:,1:])
	# print(distances, indices)

sy = np.zeros((len(Index),len(Index)))

# for i in range(len(Index)):
# 	for j in Index[i]:
# 		graph[i][j] = 1
# 		graph[j][i] = 1
# 		print(j)
# 	print(i)

#sy = Symmetric Neighborhood Set
for i in range(len(Index)):
    for j in Index[i]:
        sy[i, j] = 1
        sy[j, i] = 1

knn_matrix = np.zeros((len(Index),len(Index)))

for i in range(len(Index)):
	for j in Index[i]:
		knn_matrix[i][j] = 1

# print(knn_matrix.shape)
mut = np.zeros_like(knn_matrix)
#mut = Mutual Neighborhood Set
for i in range(10):
    ones = np.where(knn_matrix[i] == 1)
    for j in ones[0]:
        if knn_matrix[j][i] == 1:
            mut[i][j] = 1

counts = np.sum(mut == 1, axis=1)
weight = counts / k
print(weight)