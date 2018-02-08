#
# Created on Jan. 2018
# @author: Qiankun Liu
#
import numpy as np
import matplotlib.pyplot as plt
import random
import heapq
from sklearn.decomposition import PCA
from collections import Counter

def knn(folds,data,distance_type):
    if folds: # when use fold cross validation
        # knn with cross validation
        num_folds = 5
        data_size = np.shape(data)
        label = data[:,0]
        data_att = data[:,1:]# the attributes of samples in the dataset

        num_per_fold = data_size[0] / num_folds  # the number of samples in each fold
        k_range = np.arange(1, data_size[0] - num_per_fold)

        accuracy = np.zeros(np.shape(k_range))
        for k in k_range:
            accuracy_per_fold = np.zeros(num_folds)
            for fold_idx in range(0, num_folds):
                if fold_idx == num_folds - 1:
                    validation_idx = np.arange(fold_idx * num_per_fold, data_size[0])
                    
                else:
                    validation_idx = np.arange(fold_idx * num_per_fold, (fold_idx + 1) * num_per_fold)

                training_idx1 = np.arange(0, validation_idx[0])
                training_idx2 = np.arange(validation_idx[len(validation_idx) - 1] + 1, data_size[0])
                training_idx = np.append(training_idx1, training_idx2)

                test_data = data_att[validation_idx, :]
                test_label = label[validation_idx]

                training_data = data_att[training_idx, :]
                training_label = label[training_idx]

                #predict_label, accuracy_per_fold[fold_idx] = predict(k, training_data, training_label, test_data, test_label)

                predict_label = np.zeros(np.shape(test_label))
                for i in np.arange(0, len(test_label)):
                    test_sample = test_data[i, :]
                    distance = np.sqrt(np.sum(np.square(training_data - test_sample), axis=1))
                    k_neighbour = np.argsort(distance)[0:k]
                    if distance_type == 'abs':
                        distance = np.sum(np.abs(training_data - test_sample), axis=1)
                        k_neighbour = np.argsort(distance)[0:k]
                    elif distance_type == 'cos':
                        magnitude_tra = np.sqrt(np.sum(np.square(training_data),axis=1))
                        magnitude_tes = np.sqrt(np.sum(np.square(test_sample)))
                        distance = np.sum(np.multiply(training_data, test_sample),axis=1) / (magnitude_tes * magnitude_tra)
                        k_neighbour = np.argsort(distance)[len(distance)-k:len(distance)]
                        #print k_neighbour,k
                    #k_neighbour = np.argsort(distance)[0:k]
                    neighbour_label = training_label[k_neighbour]
                    predict_label[i] = Counter(neighbour_label).most_common()[0][0]

                num_right = np.sum((predict_label - test_label) == 0)
                accuracy_per_fold[fold_idx] = num_right / float(len(test_label))

            accuracy[k - 1] = np.mean(accuracy_per_fold)
    else: # when cross validation are not allowed
        iterations = 20
        data_size = np.shape(data)
        label = data[:, 0]
        data_att = data[:, 1:]  # the attributes of samples in the dataset

        num_test = data_size[0] / 5  # choose some samples as the test data
        k_range = np.arange(1, data_size[0] - num_test)

        accuracy = np.zeros(np.shape(k_range))
        for k in k_range:
            accuracy_per_itr = np.zeros(iterations)
            for itr in range(0, iterations):
                sample_idx = np.arange(0, data_size[0])
                random.shuffle(sample_idx)
                validation_idx = sample_idx[0:num_test] # for each iteration, choose the test and training data randomly
                training_idx = sample_idx[num_test:]

                test_data = data_att[validation_idx, :]
                test_label = label[validation_idx]

                training_data = data_att[training_idx, :]
                training_label = label[training_idx]

                predict_label = np.zeros(np.shape(test_label))
                for i in np.arange(0, len(test_label)):
                    test_sample = test_data[i, :]
                    distance = np.sqrt(np.sum(np.square(training_data - test_sample), axis=1))
                    k_neighbour = np.argsort(distance)[0:k]
                    if distance_type == 'abs':
                        distance = np.sum(np.abs(training_data - test_data), axis=1)
                        k_neighbour = np.argsort(distance)[0:k]
                    elif distance_type == 'cos':
                        magnitude_tra = np.sqrt(np.sum(np.square(training_data),axis=1))
                        magnitude_tes = np.sqrt(np.sum(np.square(test_sample)))
                        distance = np.sum(np.multiply(training_data, test_sample),axis=1) / (magnitude_tes * magnitude_tra)
                        k_neighbour = np.argsort(distance)[len(distance) - k:len(distance)]

                    neighbour_label = training_label[k_neighbour]
                    predict_label[i] = Counter(neighbour_label).most_common()[0][0]

                num_right = np.sum((predict_label - test_label) == 0)
                accuracy_per_itr[itr] = num_right / float(len(test_label))

            accuracy[k - 1] = np.mean(accuracy_per_itr)
    return accuracy