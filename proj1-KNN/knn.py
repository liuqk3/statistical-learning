#
# Created on Jan. 2018
# @author: Qiankun Liu
#
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from utils import *

data_origin = np.genfromtxt('.\data\wine.data',delimiter = ',')

# sorte the data randomly
data_size = np.shape(data_origin)
sample_idx = np.arange(0,data_size[0])
random.shuffle(sample_idx)
data_origin = data_origin[sample_idx,:]
accuracy_origin_folds_euc = knn(folds=True, data=data_origin, distance_type='Euclidean')
# sort_idx = np.argsort(accuracy_origin_folds_euc)
# k_optimal = sort_idx[len(sort_idx)-1]
# print k_optimal
accuracy_origin_folds_abs = knn(folds=True, data=data_origin, distance_type='abs')
accuracy_origin_folds_cos = knn(folds=True, data=data_origin, distance_type='cos')
accuracy_origin_nonfolds = knn(folds=False, data=data_origin, distance_type='Euclidean')


# standardization the data
label = np.array(data_origin[:,0])
data_att = data_origin[:, 1:]
data_att = data_att / data_att.max(0)
data_att = data_att - data_att.mean(0)
data_s = np.zeros(data_size)
data_s[:,0] = label
data_s[:,1:] = data_att
accuracy_standardize_folds_euc = knn(folds=True, data=data_s,distance_type='Euclidean')
accuracy_standardize_folds_abs = knn(folds=True, data=data_s, distance_type='abs')
accuracy_standardize_folds_cos = knn(folds=True, data=data_s, distance_type='cos')
accuracy_standardize_nonfolds = knn(folds=False, data=data_s, distance_type='Euclidean')

# PCA to dimension 6
pca = PCA(n_components=6)
data_att_pca = pca.fit_transform(data_s[:,1:])
data_pca = np.zeros((data_size[0], 6 + 1))
data_pca[:,0] = label
data_pca[:,1:] = data_att_pca
accuracy_pca_folds_euc = knn(folds=True, data=data_pca,distance_type='Euclidean')
accuracy_pca_folds_abs = knn(folds=True, data=data_pca, distance_type='abs')
accuracy_pca_folds_cos = knn(folds=True, data=data_pca, distance_type='cos')
accuracy_pca_nonfolds = knn(folds=False, data=data_pca, distance_type='Euclidean')

# T-SNE
tsne = TSNE(n_components=6)
data_att_tsne = tsne.fit_transform(data_s[:,1:])
data_tsne = np.zeros((data_size[0], 6 + 1))
data_tsne[:,0] = label
data_tsne[:,1:] = data_att_tsne
accuracy_tsne_folds_euc = knn(folds=True, data=data_tsne,distance_type='Euclidean')
accuracy_tsne_folds_abs = knn(folds=True, data=data_tsne, distance_type='abs')
accuracy_tsne_folds_cos = knn(folds=True, data=data_tsne, distance_type='cos')
accuracy_tsne_nonfolds = knn(folds=False, data=data_tsne, distance_type='Euclidean')

#visualize the data
# box plot for each attribute
plt.figure()
plt.boxplot(data_origin[:,1:])
plt.xlabel('Attributes')
plt.ylabel('values')
plt.title('Box plot of raw data')

plt.figure()
plt.boxplot(data_s[:,1:])
plt.xlabel('Attributes')
plt.ylabel('Normalized values')
plt.title('Box plot of standardized data')

# pca to show in a plane (2 dimension)
pca = PCA(n_components=2,copy=True)
data_pca = pca.fit_transform(data_s[:,1:])
# get the indice of samples
class1_idx = [idx for idx, class_label in enumerate(label) if class_label==1]
class2_idx = [idx for idx, class_label in enumerate(label) if class_label==2]
class3_idx = [idx for idx, class_label in enumerate(label) if class_label==3]
plt.figure()
plt.scatter(data_pca[class1_idx,0], data_pca[class1_idx,1], color='red')
plt.scatter(data_pca[class2_idx,0], data_pca[class2_idx,1], color='green')
plt.scatter(data_pca[class3_idx,0], data_pca[class3_idx,1], color='blue')
plt.title('Standardized data distribution after PCA (2 dimensionality)')


plt.figure()
plt.plot(accuracy_origin_folds_euc,color = 'red')
plt.plot(accuracy_standardize_folds_euc,color = 'green')
plt.plot(accuracy_pca_folds_euc,color = 'blue')
plt.plot(accuracy_tsne_folds_euc,color = 'cyan')
plt.legend(['raw data','standardized data', 'PCA (6 dimensionality)','TSNE (6 dimensionality)'])
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy with Euclidean distance (using 5-fold cross validatin)')


plt.figure()
plt.plot(accuracy_origin_folds_abs,color = 'red')
plt.plot(accuracy_standardize_folds_abs,color = 'green')
plt.plot(accuracy_pca_folds_abs,color = 'blue')
plt.plot(accuracy_tsne_folds_abs,color = 'cyan')
plt.legend(['raw data','standardized data', 'PCA (6 dimensionality)','TSNE (6 dimensionality)'])
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy with Manhattan distance (using 5-fold cross validation)')

plt.figure()
plt.plot(accuracy_origin_folds_cos,color = 'red')
plt.plot(accuracy_standardize_folds_cos,color = 'green')
plt.plot(accuracy_pca_folds_cos,color = 'blue')
plt.plot(accuracy_tsne_folds_cos,color = 'cyan')
plt.legend(['raw data','standardized data', 'PCA (6 dimensionality)','TSNE (6 dimensionality)'])
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy with cosine distance (using 5-fold corss validation)')

plt.figure()
plt.plot(accuracy_origin_nonfolds,color = 'red')
plt.plot(accuracy_standardize_nonfolds,color = 'green')
plt.plot(accuracy_pca_nonfolds,color = 'blue')
plt.plot(accuracy_tsne_nonfolds,color = 'cyan')
plt.legend(['raw data','standardized data', 'PCA (6 dimensionality)','TSNE (6 dimensionality)'])
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy with Euclidean distance (not using corss validation)')

plt.show()