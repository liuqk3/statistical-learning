#
# Created on Jan. 2018
# @author: Qiankun Liu
#
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from utils import *

# load data
xtrain = np.genfromtxt('.\data\\xtrain.txt', delimiter=',')
ctrain = np.genfromtxt('.\data\ctrain.txt')
xtest = np.genfromtxt('.\data\\xtest.txt', delimiter=',')
ptest = np.genfromtxt('.\data\ptest.txt')
c1test = np.genfromtxt('.\data\c1test.txt')

#num_att = np.shape(xtrain)[1]  # the number of attributes

################################### multi-class classification #####################################

# we first show the dstribution of raw data
c0_idx = [i for i, lab in enumerate(ctrain) if lab == 0]
c1_idx = [i for i, lab in enumerate(ctrain) if lab == 1]
plt.figure()
plt.scatter(xtrain[c0_idx ,0] ,xtrain[c0_idx ,1], color = 'red')
plt.scatter(xtrain[c1_idx ,0] ,xtrain[c1_idx ,1], color = 'green')
plt.legend(['class 0', 'class 1'])
plt.title('the distribution of raw data')

# in multi-class, multi-class 0,1 is divided from class 0, and multi-class 2,3 is divided from class 1
num_cluster = 4 # even number is better
kmeans = KMeans(n_clusters= int( num_cluster /2))
multi_label0 = kmeans.fit_predict(xtrain[c0_idx]) # divide class 0 into num_cluster/2 classes
multi_label1 = kmeans.fit_predict(xtrain[c1_idx]) # divide class 0 into num_cluster/2 classes
# change classes divided from class 1 into multi-calss 2,3
idx2 = [i for i, lab in enumerate(multi_label1) if lab == 0]
idx3 = [i for i, lab in enumerate(multi_label1) if lab == 1]
multi_label1[idx2] = 2
multi_label1[idx3] = 3
multi_label = np.append(multi_label0, multi_label1)


# show multi-calss distribution of data
plt.figure()
for idx in np.arange(0 ,num_cluster):
    cla_idx = [i for i, lab in enumerate(multi_label) if lab == idx]
    color = "#%06x" % random.randint(0, 0xFFFFFF)
    plt.scatter(xtrain[cla_idx ,0] ,xtrain[cla_idx ,1] ,color = color)
plt.legend(['multi-class 0', 'multi-class 1', 'multi-class 2', 'multi-class 3'])
plt.title('the distribution of multi-calss data generated from raw data')



######################### using linear regression ################################
method = 'ridge' # 'ridge', 'lasso', 'linear'
order = np.arange(1,30)#8 # the order of polynomials
#lamb = [1]
lamb = np.arange(0,30,0.1)
#lamb = (0,np.exp(-5),np.exp(0),np.exp(5),np.exp(10),np.exp(20),np.exp(50),np.exp(100))
error = np.zeros((len(lamb),len(order)))
for idx_ord in range(0,len(order)):
    for idx_lamb in range(0,len(lamb)):#len(lamb)):

        pre_label = 2 * np.ones((np.shape(xtest)[0],))  # used to store the predicted label (class 0 and 1)
        # make some copies of data for multi-classification
        multi_label_tmp = np.copy(multi_label)
        xtest_tmp = np.copy(xtest)
        xtrain_tmp = np.copy(xtrain)
        tra_label_tmp = np.copy(multi_label)

        for clu_idx in range(0, num_cluster):
            # train
            tra_c1_idx = [i for i, lab in enumerate(multi_label_tmp) if lab == clu_idx]
            tra_c0_idx = [i for i, lab in enumerate(multi_label_tmp) if lab != clu_idx]
            tra_label_tmp[tra_c0_idx] = 0
            tra_label_tmp[tra_c1_idx] = 1
            tra_basic_fun = get_basic_function(xtrain_tmp, order[idx_ord])
            w = get_linear_solution(tra_basic_fun, tra_label_tmp,lamb[idx_lamb],method)
            # test
            tes_basic_fun = get_basic_function(xtest_tmp,order[idx_ord])
            pre_label_tmp = linear_predict(tes_basic_fun,w)
            pre_c1_idx = [i for i, lab in enumerate(pre_label_tmp) if lab == 1]
            pre_label[pre_c1_idx] = clu_idx

            #adjust for next precessing, remove the classified class
            multi_label_tmp = multi_label_tmp[tra_c0_idx]
            tra_label_tmp = tra_label_tmp[tra_c0_idx]
            xtrain_tmp = xtrain_tmp[tra_c0_idx,:]
            xtest_tmp = xtest_tmp[tra_c0_idx,:]

        # convert the multi-class results in to binary classes
        c0_idx = [i for i, lab in enumerate(pre_label) if lab == 0 or lab == 1]
        c1_idx = [i for i, lab in enumerate(pre_label) if lab == 2 or lab == 3]
        pre_label[c0_idx] = 0
        pre_label[c1_idx] = 1

        #print(len(c0_idx),len(c1_idx))

        ground_truth = np.vstack((ptest,c1test))
        error[idx_lamb,idx_ord] = error_rate(pre_label,ground_truth, 'test')

fig = plt.figure()
ax = Axes3D(fig)
y = lamb#np.meshgrid(lamb)
x = order#np.meshgrid(order)
x,y = np.meshgrid(x,y)
ax.plot_surface(x,y,error,cmap='rainbow')
ax.set_xlabel('order of polynomials')
ax.set_ylabel('λ')
ax.set_zlabel('error rate')
ax.set_title('error rate respective to λ and order of polynomials')

print(np.shape(error))
min_error = np.min(error)
location = np.where(error == min_error)
optimal_lamb = lamb[location[0]]
optimal_order = order[location[1]]
print('The optimal lambda and order are (', optimal_lamb, ',', optimal_order,') and the error rate is ',min_error )

plt.show()


