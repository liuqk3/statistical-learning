
# Created on Jan. 2018
# @author: Qiankun Liu

from __future__ import print_function
import numpy as np
from utils import *
import os
import matplotlib.pyplot as plt

data_path = './data/proj4/'
catagories = np.array(os.listdir(data_path))
cata_idx = np.arange(0, len(catagories)-1)
distance_type = 'Euclidean'# 'Euclidean', 'Manhattan', 'cosine'
model_name = 'resnet34'

############################# show the evaluation when k=10,20,50 and 100 ####################################
k_required = np.array([10,20,50,100]) # the number of nearest k neighbour
for k_idx in  np.arange(0,len(k_required)):
    k = k_required[k_idx]
    retrival = np.load('./results/retrival_' + model_name + '_'+distance_type + '_' + str(k) + '.npy')
    pk = evaluation(retrival,catagories,'P@K')
    pk_mean = '%.3f'%np.mean(pk)

    rk = evaluation(retrival, catagories,'R@K')
    rk_mean = '%.3f'%np.mean(rk)

    fk = evaluation(retrival, catagories, 'F@K')
    fk_mean = '%.3f'%np.mean(fk)

    mrrk = evaluation(retrival,catagories,'MRR@K')
    mrrk_mean = '%.3f'%np.mean(mrrk)

    plt.figure()
    plt.plot(cata_idx,pk,color = 'red')
    plt.scatter(cata_idx, pk,color='red')

    plt.plot(cata_idx,rk,color = 'green')
    plt.scatter(cata_idx, rk,color='green')

    plt.plot(cata_idx,fk,color = 'blue')
    plt.scatter(cata_idx, fk,color = 'blue')

    plt.plot(cata_idx,mrrk, color='orange')
    plt.scatter(cata_idx, mrrk, color='orange')

    plt.legend(['P@K, average: '+str(pk_mean),'R@K, average: '+str(rk_mean),
                'F@K, average: '+str(fk_mean),'MRR@K, average: '+str(mrrk_mean)],
               fontsize = 9)# loc = 'upper center')
    plt.xlabel('category')
    plt.ylabel('evaluation values')
    plt.title('evaluation on image retrieval (k='+str(k)+', '+distance_type+' distance, '+model_name +')')
    plt.savefig('./figures/'+model_name+'_'+distance_type+'_'+str(k)+'.pdf')

############################# show the pk, rk, fk, mrrk curve trend versus k ##########################################
k_range = np.arange(1,120)
k_required = np.array([10,20,50,100])
eva = np.zeros((4,len(k_range)))
for k_idx in  np.arange(0,len(k_range)):
    k = k_range[k_idx]
    retrival = np.load('./results/retrival_' + model_name + '_'+distance_type + '_' + str(k) + '.npy')
    pk = evaluation(retrival,catagories,'P@K')
    pk_mean = '%.3f'%np.mean(pk)

    rk = evaluation(retrival, catagories,'R@K')
    rk_mean = '%.3f'%np.mean(rk)

    fk = evaluation(retrival, catagories, 'F@K')
    fk_mean = '%.3f'%np.mean(fk)

    mrrk = evaluation(retrival,catagories,'MRR@K')
    mrrk_mean = '%.3f'%np.mean(mrrk)

    eva[0,k_idx] = pk_mean
    eva[1,k_idx] = rk_mean
    eva[2,k_idx] = fk_mean
    eva[3,k_idx] = mrrk_mean


plt.figure()
plt.plot(k_range,eva[0,:],color = 'red')
plt.scatter(k_required, eva[0,k_required-1],color='red')

plt.plot(k_range,eva[1,:],color = 'green')
plt.scatter(k_required, eva[1,k_required-1],color='green')

plt.plot(k_range,eva[2,:],color = 'blue')
plt.scatter(k_required, eva[2,k_required-1],color = 'blue')

plt.plot(k_range,eva[3,:], color='orange')
plt.scatter(k_required, eva[3,k_required-1], color='orange')

plt.legend(['P@K','R@K','F@K','MRR@K'],fontsize = 9)# loc = 'upper center')
plt.xlabel('k')
plt.ylabel('evaluation values')
plt.title('average evaluation on image retrieval ('+distance_type+' distance, '+model_name +')')
plt.savefig('./figures/'+model_name+'_'+distance_type+'_'+str(k)+'_average.pdf')


plt.show()