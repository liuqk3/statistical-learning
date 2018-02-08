
# created on Jan. 2018
# @author Qiankun Liu

import numpy as np
import time
from utils import *

# load data
movies = np.genfromtxt('.\data\\task1\movies.dat',dtype=str, delimiter='::',comments='****')#, deletechars=';')
ratings = np.genfromtxt('.\data\\task1\\ratings.dat',dtype=str, delimiter='::')
num_folds = 10
for fold_idx in np.arange(0,num_folds):
    file_name = '.\data\\task1\\users.dat' + str(fold_idx)
    if fold_idx == 0:
        users = np.genfromtxt(file_name, dtype=str, delimiter='::')
    else:
        users_tmp = np.genfromtxt(file_name, dtype=str, delimiter='::')
        users = np.dstack((users, users_tmp))

# cross validation
error_gen = np.zeros((num_folds,))
time_gen = np.zeros((num_folds,2)) # first column is training time, second column is test time
error_age = np.zeros((num_folds,))
time_age = np.zeros((num_folds,2)) # first column is training time, second column is test time

for fold in np.arange(0,num_folds):
    print('fold = ', fold, '\n')
    test_users = users[:,:,fold]
    train_users = []
    train_idx1 = np.arange(0,fold)
    train_idx2 = np.arange(fold + 1, num_folds)
    train_idx = np.append(train_idx1,train_idx2)
    for idx in np.arange(0,len(train_idx)):
        if idx == 0:
            train_users = users[:,:,train_idx[idx]]
        else:
            train_users = np.vstack((train_users,users[:,:,train_idx[idx]]))

    ################################# predict gender ####################################
    t1_train = time.time()
    prob_gen = probability(movies,ratings,train_users,fold, 'gender')
    # file_name = './cache/prob_gender_' + str(fold) + '.npy'
    # prob_gen = np.load(file_name)
    t2_train = time.time()

    t1_test = time.time()
    predict_gen = predict(movies,ratings,train_users,test_users,prob_gen, 'gender')
    t2_test = time.time()

    error_gen[fold] = error_rate(predict_gen,test_users[:,1], 'gender')

    time_gen[fold,0] = t2_train - t1_train
    time_gen[fold,1] = t2_test - t1_test

    ################################# predict gender ####################################
    t1_train = time.time()
    prob_age = probability(movies,ratings,train_users,fold, 'age')
    # file_name = './cache/prob_age_' + str(fold) + '.npy'
    # prob_age = np.load(file_name)
    t2_train = time.time()

    t1_test = time.time()
    predict_age = predict(movies,ratings,train_users,test_users,prob_age, 'age')
    t2_test = time.time()

    error_age[fold] = error_rate(predict_age, test_users[:, 2], 'age')

    time_age[fold,0] = t2_train - t1_train
    time_age[fold,1] = t2_test - t1_test

np.save('./cache/time_gender.npy',time_gen)
np.save('./cache/error_gender.npy',error_gen)
print(time_gen,'\n')
print(error_gen)

np.save('./cache/time_age.npy',time_age)
np.save('./cache/error_age.npy',error_age)
print(time_age, '\n')
print(error_age)

