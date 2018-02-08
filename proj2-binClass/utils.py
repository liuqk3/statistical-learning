#
# Created on Jan. 2018
# @author: Qiankun Liu
#

import numpy as np
import math
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
from decimal import *
from sklearn import linear_model

def get_basic_function(data, order):
    # this function return the polynomial function as the basic function
    # the basis of learned model must be considered, so the actually order
    # is (order + 1)
    #
    # data: a (N,2) array
    # order:integer
    data_shape = np.shape(data)
    basic_function = np.zeros((data_shape[0], data_shape[1] * (order + 1))) # (N, 2*(order + 1)) array
    for ord in np.arange(0, order+1):
        basic_function[:,2*ord:2*ord + 1 + 1] = np.power(data,ord)
    return basic_function

def error_rate(pre_label, ground_truth, mode):
    # this function return the error rate for linear regression
    #
    # pre_label: (N,) array, the predicted label of test samples
    # mode: string and it can be 'validation' or 'test'
    # ground_truth: in the mode 'validation', it is a (N,) array containing the label of training samples
    #               in the mode 'test', it is a (2,N) array, the first and rows are the ptest, c1test respectively
    if mode == 'test':
        ptest = ground_truth[0,:]
        c1test = ground_truth[1,:]
        pre_c1_idx = [i for i, pre in enumerate(pre_label) if pre == 1] # the index of samples that predicted to be class 1
        pre_c0_idx = [i for i, pre in enumerate(pre_label) if pre != 1] # the index of samples that predicted to be class 0
        error1 = np.dot(ptest[pre_c0_idx], c1test[pre_c0_idx])
        error2 = np.dot(ptest[pre_c1_idx], np.subtract(1, c1test[pre_c1_idx]))
        error = error1 + error2
    elif mode == 'validation':
        ground_truth_tmp = np.zeros(np.shape(ground_truth))
        ground_truth_tmp[:] = ground_truth # we make a copy of ground_truth
        c0_idx = [i for i, lab in enumerate(ground_truth) if lab <= 0]
        ground_truth_tmp[c0_idx] = -1
        num_right = np.sum(ground_truth_tmp - pre_label == 0)
        error = 1 - num_right / float(len(ground_truth_tmp))

    return error

#######################################################################################
#                                                                                     #
#                Here we define some functions for linear regression                  #
#                                                                                     #
#######################################################################################

def get_linear_solution(basic_function, label, regularizer, regression_type):
    # this function return the weights of model, which are trined from the
    # basic function and this function use linear regression
    #
    # basic_function: (N, 2*(order + 1)) array
    # label: (N,) array, the label of training data
    # regularizer: the regularization
    # regression_type: string, 'ridge', 'lasso', 'lda'

    # we first let -1 denotes class 0, and 1 denotes class 1
    label_tmp = np.zeros(np.shape(label))
    label_tmp[:] = label # we make a copy of label
    c0_idx = [i for i, lab in enumerate(label_tmp) if lab == 0]
    label_tmp[c0_idx] = -1
    basic_shape = np.shape(basic_function)
    if regression_type == 'lasso': # Lasso regression
        clf = linear_model.Lasso(alpha=regularizer)
        clf.fit(basic_function, label)
        w = clf.coef_
    elif regression_type == 'ridge': # linear regression
        I = np.eye(basic_shape[1])
        A = regularizer * I + np.dot(np.transpose(basic_function),basic_function)
        B = np.dot(np.transpose(basic_function), label_tmp)
        # if np.linalg.det(A) == 0: # if A can not be inversed, we compute the pseudoinverse
        #     w = np.dot(np.linalg.pinv(A),B) # W is a (2*(order+1),) array
        # else:
        #     w = np.dot(np.linalg.inv(A), B)  # W is a (2*(order+1),) array
        w = np.dot(np.linalg.pinv(A), B)  # W is a (2*(order+1),) array
    elif regression_type == 'lda':
        c1_idx = [i for i, lab in enumerate(label_tmp) if lab == 1]
        m0 = np.mean(basic_function[c0_idx,:], axis=0)
        m1 = np.mean(basic_function[c1_idx,:], axis=0)
        sw0 = np.dot(np.transpose(basic_function[c0_idx,:] - m0),basic_function[c0_idx,:] - m0)
        sw1 = np.dot(np.transpose(basic_function[c1_idx,:] - m1),basic_function[c1_idx,:] - m1)
        sw = sw0 + sw1
        w = np.dot(np.linalg.pinv(sw), m1 - m0)
    return w

def linear_predict(data, weight):
    # this function return the predicted label by linear regression classifier
    #
    # data: (N, 2*(order + 1)) array, the data that has been changed to basic function form
    # weight: (2*(order + 1),) array, the weigths of model
    pre_label = np.dot(data, weight)
    pre_c1_idx = [i for i, pre in enumerate(pre_label) if pre >= 0] # the index of samples that predicted to be class 1
    pre_c0_idx = [i for i, pre in enumerate(pre_label) if pre < 0] # the index of samples that predicted to be class 0
    pre_label[pre_c1_idx] = 1
    pre_label[pre_c0_idx] = -1
    return pre_label

#######################################################################################
#                                                                                     #
#               Here we define some functions for logistic regression                 #
#                                                                                     #
#######################################################################################
def get_logistic_solution(basic_function, label, method):
    # this function return the weights of model using logistic regression
    #
    # basic_function: (N, 2*(order + 1)) array
    # label: (N,) array, the label of training data
    # method: string and it can be 'ml', 'newton_raphson'

    if method == 'ml': # maximum likelihood method
        num_samples = np.shape(basic_function)[0]
        mu = np.mean(basic_function, axis=0)
        sigma = float(num_samples) / (num_samples - 1) * np.dot(np.transpose(basic_function - mu), (basic_function - mu))

        c1_idx = [i for i, lab in enumerate(label) if lab == 1]
        c0_idx = [i for i, lab in enumerate(label) if lab == 0]
        # label[c0_idx] = 0
        c1 = basic_function[c1_idx,:]
        c0 = basic_function[c0_idx,:]
        mu1 = np.mean(c1,axis=0)
        mu0 = np.mean(c0,axis=0)
        # if np.linalg.det(sigma) == 0:
        #     w = np.dot(np.linalg.pinv(sigma),(mu1 - mu0))
        # else:
        #     w = np.dot(np.linalg.inv(sigma),(mu1 - mu0))
        w = np.dot(np.linalg.pinv(sigma), (mu1 - mu0))
    elif method == 'newton_raphson':
        iterations = 5000
        w = np.zeros(np.shape(basic_function)[1]) # initial the weight to zeros
        for itr in np.arange(0,iterations):
            pre_prob = logistic_predict(basic_function, w, 'train')
            R = np.diag(pre_prob * (1 - pre_prob)) # (N,N) array
            H = np.dot(np.dot(np.transpose(basic_function), R), basic_function) # (2*(order + 1), 2*(order + 1)) array
            # if np.linalg.det(H) == 0:
            #     H_inv = np.linalg.pinv(H)
            # else:
            #     H_inv = np.linalg.inv(H)
            H_inv = np.linalg.pinv(H)
            w = w - np.dot(np.dot(H_inv, np.transpose(basic_function)), (pre_prob - label))
    elif method == 'gradient_descent':
        yita = 0.001 # the learning rate
        iterations = 5000
        w = np.zeros(np.shape(basic_function)[1]) # initial the weights
        for itr in np.arange(0,iterations):
            pre_prob = logistic_predict(basic_function, w, 'train')
            gradient = np.sum(np.transpose(np.transpose(basic_function) * (pre_prob - label)),axis=0)
            w = w - yita * gradient
    #print(w)
    return w

def logistic_predict(data, weight, mode):
    # this function return the predicted label by logistic regression classifier
    #
    # data: (N, 2*(order + 1)) array, the data that has been changed to basic function form
    # weight: (2*(order + 1),) array, the weigths of model
    # mode: string and it can be 'validation' or 'test'
    pre_prob = np.dot(data, weight)
    pre_prob = 1.0 / (1.0 + np.power(np.exp(1), -pre_prob))
    if mode == 'train':
        return pre_prob
    elif mode == 'test':
        pre_label = np.zeros(np.shape(pre_prob))
        c1_idx = [i for i, prob in enumerate(pre_prob) if prob >= 0.5]
        c0_idx = [i for i, prob in enumerate(pre_prob) if prob < 0.5]
        pre_label[c1_idx] = 1
        pre_label[c0_idx] = -1
        return pre_label


def bayesian_information_criterion(pre_prob, ground_truth, order):
    # this function return the error rate for logistic regression
    #
    # pre_label: (N,) array, the predicted label of test samples
    # ground_truth: in the mode 'validation', it is a (N,) array containing the label of training samples
    #               in the mode 'test', it is a (2,N) array, the first and rows are the ptest, c1test respectively
    # order: the order of polynomials, so there will be order + 1 numbers in the learned model

    bic = 1
    for pre_prob_idx in np.arange(0,len(ground_truth)):
        if ground_truth[pre_prob_idx] == 0:
            bic = bic * (1 - pre_prob[pre_prob_idx])
        else:
            bic = bic * pre_prob[pre_prob_idx]
    #print(bic)
    bic = bic / np.power(np.sqrt(len(pre_prob)), 2 * (order + 1))# bic = math.log(bic) - 0.5 * 2 *(order + 1) * math.log(len(pre_prob))
    return bic