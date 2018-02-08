
# Created on Jan. 2018
# @author: Qiankun Liu

#from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms, models

def deep_features(model, image):
    # this function extrac the feature from the input image using model
    # input:
    # model: the CNN model used to extract deeft feature
    #image: a image, opend by PIL.Image.open() function
    # output:
    # out: a 1-D array, the output of model

    # define the transform for data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    image = image.resize((224, 224))
    image = data_transform(image)
    # print(image)
    image = np.expand_dims(image,0)
    image = Variable(torch.from_numpy(image))
    out = model(image)
    out = out.data.numpy()[0] # transform to numpy
    return out


def get_similar_images(catagories, query_idx, lib, k, distanceType):
    # this function can perform image retrieval
    # input:
    # categories: (41,) array, each element is a string denoting the category
    # query_idx: integet, denotes the query image
    # lib: 2-D array, each row in lib is the feature extracted from a image
    # k: interger, the top-k similar images will be returned
    # distanceType: string, determine the distance type to use, it can be
    #    'Euclidean', 'Manhattan', 'cosine'
    # output:
    # k_similar: the indices of top-k similar images

    query = lib[query_idx]
    if distanceType == 'Euclidean':
        differenc = lib - query
        distance = np.sqrt(np.sum(np.square(differenc),axis=1)) # Euclidean distance is default
    if distanceType == 'Manhattan':
        differenc = lib - query
        distance = np.sum(np.abs(differenc),axis=1)
    elif distanceType == 'cosine':
        magnitude_query = np.sqrt(np.sum(np.square(query)))
        magnitude_lib = np.sqrt(np.sum(np.square(lib),axis=1))
        distance = - np.sum(np.multiply(lib, query),axis=1) / (magnitude_lib * magnitude_query)

    sort_idx = np.argsort(distance)
    similar = sort_idx[0:k]
    if query_idx in similar:
        similar = np.delete(similar, np.where(similar == query_idx)[0][0])
        similar = np.append(similar, sort_idx[k])
    k_similar = np.array([])
    for idx in np.arange(0,k):
        similar_idx = similar[idx]
        cata_idx = similar_idx / 50
        # cata = catagories[cata_idx]
        # im_name = 'image_'+ str('%04d'%(similar_idx%50 + 1)) + '.jpg'
        # one_similar = {}
        # one_similar['catagory'] = cata
        # one_similar['image_name'] = im_name
        # k_similar = np.append(k_similar, one_similar)
        k_similar = np.append(k_similar, cata_idx)
    return k_similar

def evaluation(retrival, catagories, type):
    # this function can evaluate the retrieval results
    # input:
    # retrival: 2-D array, each row is the top-k indices of the image index by this row-index
    # catagories: (41,) array, each element is a string denoting the category
    # type: #string, it can be 'P@K', 'R@K','F@K','MRRK@K'

    eva = np.zeros((len(catagories),))
    for cata_idx in np.arange(0, len(catagories)):
        retrival_tmp = np.copy(retrival[50 * cata_idx:50 * (cata_idx + 1),:])
        mask = retrival_tmp == cata_idx
        retrival_tmp[mask] = 1
        retrival_tmp[~mask] = 0
        if type == 'P@K':
            pk_per_im = np.sum(retrival_tmp, axis=1) / float(np.shape(retrival)[1])
            pk = np.mean(pk_per_im)
            eva[cata_idx] = pk
        if type == 'R@K':
            rk_per_im = np.sum(retrival_tmp, axis=1) / float(49) # remove the query image from the catagory
            rk = np.mean(rk_per_im)
            eva[cata_idx] = rk
        if type == 'F@K':
            pk_per_im = np.sum(retrival_tmp, axis=1) / float(np.shape(retrival)[1])
            pk = np.mean(pk_per_im)
            rk_per_im = np.sum(retrival_tmp, axis=1) / float(49)
            rk = np.mean(rk_per_im)
            fk = 2.0 / (1.0/pk + 1.0/rk)
            eva[cata_idx] = fk
        if type == 'MRR@K':
            mrrk_per_im = np.zeros((50,))
            for im_idx in np.arange(0,50):
                num_right = np.sum(retrival_tmp[im_idx,:])
                if num_right == 0:
                    mrrk_per_im[im_idx] = 0
                else:
                    right_idx = np.where(retrival_tmp[im_idx,:] == 1)[0]
                    mrrk_per_im[im_idx] = np.sum(1.0 / (right_idx + 1)) / float(num_right)
            mrrk = np.mean(mrrk_per_im)
            eva[cata_idx] = mrrk
    clutter_idx = np.where(catagories == 'clutter')[0][0]
    eva = np.delete(eva, clutter_idx)
    return eva