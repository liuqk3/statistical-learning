
# Created on Jan. 2018
# @author: Qiankun Liu

from __future__ import print_function
import numpy as np
import torch
from torchvision import models
from PIL import Image
from sklearn.decomposition import PCA
import time
import os
from utils import *

data_path = './data/proj4/'
catagories = np.array(os.listdir(data_path))
############################## extract features #############################################
model_name = 'resnet34' # used to save some results
# load model and modify it
model = models.resnet34(pretrained=True)
# for m_name, m in model.named_children():
#     print(m_name)
fc = nn.Sequential(*list(model.fc.children()))
model.fc = fc
t1 = time.time()
# extract features
for idx_cata in np.arange(0,len(catagories)):
    catagory = catagories[idx_cata]
    catagory_path = data_path + catagory
    file_names = os.listdir(catagory_path)
    for idx_im in np.arange(0,len(file_names)):
        im_name = file_names[50 - idx_im - 1] # in order to read images form 0001 to 0050
        im_path = catagory_path + '/' + im_name
        print('Processing image:', im_path)
        im = Image.open(im_path)
        if len(np.shape(im)) == 2:
            im = im.convert('RGB')
        out = deep_features(model,im)
        if idx_cata == 0 and idx_im ==0:
            features = out
        else:
            features = np.vstack((features,out))
np.save('./results/features_'+ model_name+'.npy', features)
t2 = time.time()
t_extract_feature = t2 - t1
np.save('./results/time_extract_features_'+model_name+'.npy',t_extract_feature)


# ####################### load the raw features ###############################
features = np.load('./results/features_'+ model_name+'.npy')
distance_type = 'Euclidean'# 'Manhattan' 'Euclidean' 'cosine'
k_range = np.array([10,20,50,100]) # the number of nearest k neighbour
#k_range = np.arange(1,120)
t_retrival = np.zeros((len(k_range),))
for k_idx in np.arange(0,len(k_range)):
    t1 = time.time()
    k = k_range[k_idx]
    print('Retrieval: K=',k)
    retrival = np.zeros((np.shape(features)[0],k))
    for im_idx in np.arange(0, np.shape(features)[0]):
        gt_idx = im_idx/50
        gt_cata = catagories[gt_idx]
        if gt_cata != 'clutter':
            k_similar = get_similar_images(catagories,im_idx, features, k, distance_type)
            retrival[im_idx,:] = k_similar
    t2 = time.time()
    t_retrival[k_idx] = t2 - t1
    np.save('./results/retrival_'+model_name+'_' + distance_type + '_' + str(k) + '.npy', retrival)
np.save('./results/time_retrival_' + model_name + '.npy', t_retrival)

####################### reduct the dimensionality of features using PCA ###############################
d = 50#np.shape(features)[1] / 4
pca = PCA(n_components=d)
t1 = time.time()
features_pca = pca.fit_transform(features)
t2 = time.time()
print('PCA time consuming:',t2-t1)
t_retrival = np.zeros((len(k_range),))
for k_idx in np.arange(0,len(k_range)):
    t1 = time.time()
    k = k_range[k_idx]
    retrival = np.zeros((np.shape(features_pca)[0],k))
    for im_idx in np.arange(0, np.shape(features_pca)[0]):
        gt_idx = im_idx/50
        gt_cata = catagories[gt_idx]
        if gt_cata != 'clutter':
            k_similar = get_similar_images(catagories,im_idx, features_pca, k, distance_type)
            retrival[im_idx,:] = k_similar
    t2 = time.time()
    t_retrival[k_idx] = t2 - t1
    np.save('./results/retrival_'+model_name+'_pca_' + distance_type + '_' + str(k) + '.npy', retrival)
np.save('./results/time_retrival_' + model_name + '_pca.npy', t_retrival)