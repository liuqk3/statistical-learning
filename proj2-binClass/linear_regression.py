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
from utils import *

# load data
xtrain = np.genfromtxt('.\data\\xtrain.txt',delimiter=',')
ctrain = np.genfromtxt('.\data\ctrain.txt')
xtest = np.genfromtxt('.\data\\xtest.txt',delimiter=',')
ptest = np.genfromtxt('.\data\ptest.txt')
c1test = np.genfromtxt('.\data\c1test.txt')

num_att = np.shape(xtrain)[1] # the number of attributes

##################################### linear regression ############################################
#              In the following codes, we can do the type of regression as follows                 #
#                                     ridge regression                                             #
#                                     Lasso regression                                             #
#                                       Fisher's LDA                                               #
####################################################################################################
# not using cross validation
regression_type = 'lda' # 'ridge', 'lasso', 'lda'
order = 4 # we set the order of polynomial 4
#lamb = [1e-10, np.exp(-1), np.exp(0), np.exp(1), np.exp(2.4), np.exp(10), np.exp(100)]
lamb = np.arange(0,20,0.05)

w = np.zeros((len(lamb),num_att*(order + 1)))
error = np.zeros((len(lamb),))

tra_basic_fun = get_basic_function(xtrain, order)
tes_basic_fun = get_basic_function(xtest, order)

for lamd_idx in np.arange(0,len(lamb)):
    w[lamd_idx, :] = get_linear_solution(basic_function=tra_basic_fun, label=ctrain, regularizer=lamb[lamd_idx], regression_type=regression_type)
    # test
    pre_label = linear_predict(tes_basic_fun, w[lamd_idx, :])
    test_ground_truth = np.vstack((ptest, c1test))
    error[lamd_idx] = error_rate(pre_label=pre_label, ground_truth=test_ground_truth, mode='test')

    w_ls = w[0,:]
    error_ls = error[0]

optimal_lamb = lamb[np.argsort(error)[0]]
optimal_error_lamb = error[np.argsort(error)[0]]
if regression_type != 'lda':# lda doesn't use the regularizer lambda, so we do not show this figure
    print('The optimal lambda is ', optimal_lamb, 'and the error rate is ',optimal_error_lamb )
    # show the test performance
    plt.figure(1)
    plt.plot(lamb, error)
    plt.xlabel('λ')
    plt.ylabel('error rate')
    plt.title('The error rate curve respect to λ (4 order polynomials)')
print('The least square solution is', w[0,:])

# using cross validation
num_folds = 5
num_per_fold = int(np.shape(xtrain)[0] / num_folds)
# resort train data randomly
random_index = np.arange(0, np.shape(xtrain)[0])
random.shuffle(random_index)
# compute the error rate of cross validation and test
order_r = np.arange(1,50)
val_error = np.zeros((len(order_r, )))
tes_error = np.zeros((len(order_r, )))
for idx in np.arange(0, len(order_r)):
    val_error_per_fold = np.zeros((num_folds,))
    tes_error_per_fold = np.zeros((num_folds,))
    order = order_r[idx]
    for fold_idx in np.arange(0,num_folds):
        if fold_idx == (num_folds - 1):
            validation_idx = np.arange(fold_idx * num_per_fold, np.shape(xtrain)[0])
        else:
            validation_idx = np.arange(fold_idx * num_per_fold, (fold_idx + 1) * num_per_fold)
        train_idx1 = np.arange(0,validation_idx[0])
        train_idx2 = np.arange(validation_idx[len(validation_idx) - 1], np.shape(xtrain)[0])
        train_idx = np.append(train_idx1, train_idx2)

        validation_idx = random_index[validation_idx]
        train_idx = random_index[train_idx]

        train_data = xtrain[train_idx,:]
        train_class = ctrain[train_idx]

        validation_data = xtrain[validation_idx,:]
        validation_calss = ctrain[validation_idx]
        # train
        tra_basic_fun = get_basic_function(train_data,order)
        w = get_linear_solution(tra_basic_fun, train_class,optimal_lamb, regression_type=regression_type)
        # validation
        val_basic_fun = get_basic_function(validation_data,order)
        pre_label = linear_predict(val_basic_fun, w)
        val_error_per_fold[fold_idx] = error_rate(pre_label, validation_calss, 'validation')
        # test
        tes_basic_fun = get_basic_function(xtest, order)
        pre_label = linear_predict(tes_basic_fun, w)
        test_ground_truth = np.vstack((ptest,c1test))
        tes_error_per_fold[fold_idx] = error_rate(pre_label, test_ground_truth, 'test')

    val_error[idx] = np.mean(val_error_per_fold)
    tes_error[idx] = np.mean(tes_error_per_fold)
optimal_val_order = order_r[np.argsort(val_error)[0]]
optimal_val_error = val_error[np.argsort(val_error)[0]]
optimal_tes_order = order_r[np.argsort(tes_error)[0]]
optimal_tes_error = tes_error[np.argsort(tes_error)[0]]
print('############################ linear regression #############################')
print('The optiaml validation order is ', optimal_val_order, 'and the optimal validation error rate is ', optimal_val_error)
print('The optiaml test order is ', optimal_tes_order, 'and the optimal test error rate is ', optimal_tes_error, '\n')

plt.figure()
plt.plot(order_r, val_error, color = 'red')
plt.plot(order_r, tes_error, color = 'green')
plt.legend(['validation', 'test'])
plt.title('error rate curve respect to order of polynomials')
plt.xlabel('order of polynomials')
plt.ylabel('error rate')

plt.show()

