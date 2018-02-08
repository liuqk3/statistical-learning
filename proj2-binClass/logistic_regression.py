import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
from utils import *

# load data
xtrain = np.genfromtxt('.\data\\xtrain.txt', delimiter=',')
ctrain = np.genfromtxt('.\data\ctrain.txt')
xtest = np.genfromtxt('.\data\\xtest.txt', delimiter=',')
ptest = np.genfromtxt('.\data\ptest.txt')
c1test = np.genfromtxt('.\data\c1test.txt')

num_att = np.shape(xtrain)[1]  # the number of attributes

##################################### logistic regression ##########################################
#              In the following codes, we can solve logistic regression by two methods             #
#                                      Maximum likelihood                                          #
#                                        Newton-Raphson                                            #
#                                       Gradient descent                                           #
####################################################################################################
method = 'gradient_descent' # 'ml', 'newton_raphson', 'gradient_descent'
order_r = np.arange(1,25)
tra_bic = np.zeros((len(order_r,)))
tes_error = np.zeros((len(order_r,)))
optimal_tes_error = 1
optimal_tra_bic = 0
for idx in np.arange(0, len(order_r)):
    tra_basic_fun = get_basic_function(xtrain, order_r[idx])

    w = get_logistic_solution(basic_function=tra_basic_fun, label=ctrain, method=method)
    # print(w)
    # compute train bic
    pre_prob = logistic_predict(tra_basic_fun, w, 'train')
    tra_bic[idx] = bayesian_information_criterion(pre_prob, ctrain, order_r[idx])
    if tra_bic[idx] > optimal_tra_bic:
        optimal_tra_w = w
        optimal_tra_bic = tra_bic[idx]
        optimal_tra_order = order_r[idx]


    # compute test bic
    tes_basic_fun = get_basic_function(xtest, order_r[idx])
    pre_label = logistic_predict(tes_basic_fun, w, 'test')
    test_ground_truth = np.vstack((ptest,c1test))
    tes_error[idx] = error_rate(pre_label, test_ground_truth, 'test')
    if tes_error[idx] < optimal_tes_error:
        optimal_tes_w = w
        optimal_tes_error = tes_error[idx]
        optimal_tes_order = order_r[idx]
plt.figure()
plt.plot(order_r, tra_bic )#/ np.max(tra_bic), color = 'red')
plt.xlabel('order of polymials')
plt.ylabel('raw values (not logarithmic)')
plt.title('Bayesian information criterion (' + method + ')')
plt.savefig('./results/1.pdf')

plt.figure()
plt.plot(order_r, tes_error)# , color = 'green')
#plt.legend(['normalized Baysian information criterion', 'test error'])
plt.xlabel('order of polymials')
plt.ylabel('error rate')
plt.title('error rate respect to order of polynomials (' + method + ')')

# optimal_bic_order = order_r[np.argsort(tra_bic)[len(tra_bic) - 1]] # the larger bic is, the better performance of classifier
# optimal_tra_bic = tra_bic[np.argsort(tra_bic)[len(tra_bic) - 1]]
# optimal_tes_order = order_r[np.argsort(tes_error)[0]]
# optimal_tes_error = tes_error[np.argsort(tes_error)[0]]

print('\nThe optiaml Beyesian information criterion order is ', optimal_tra_order, 'and the optimal Bayesian information criterion is ', optimal_tra_bic)
print('The optiaml test order is ', optimal_tes_order, 'and the optimal test error rate is ', optimal_tes_error, '\n')

plt.show()