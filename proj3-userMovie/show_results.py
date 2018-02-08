
# created on Jan. 2018
# @author Qiankun Liu

import numpy as np
import matplotlib.pyplot as plt

error_gen = np.load('.\cache\error_gender.npy')
error_age = np.load('.\cache\error_age.npy')
time_gen = np.load('.\cache\\time_gender.npy')
time_age = np.load('.\cache\\time_age.npy')

print(np.mean(error_gen), np.mean(error_age), np.mean(time_gen,axis=0), np.mean(time_age, axis=0))



fold_idx = np.arange(0, len(error_gen))

plt.figure()
plt.bar(left=fold_idx, height=error_gen, width=0.8)
plt.xlabel('fold')
plt.ylabel('error rate')
plt.title('error rate of prediction of gender for different folds')
for x,y in zip(fold_idx,error_gen):
    plt.text(x, y, '%.3f' % y, ha='center', va= 'bottom')

plt.figure()
plt.bar(left=fold_idx - 0.225, height=time_gen[:,0], width=0.45, facecolor = 'lightskyblue')
plt.bar(left=fold_idx + 0.225, height= time_gen[:,1], width=0.45,facecolor = 'yellowgreen')
plt.legend(['runtime of training', 'runtime of test'], loc= 'center left')
plt.xlabel('fold')
plt.ylabel('time (seconds)')
plt.title('runtime of training and test for prediction of gender')
for x,y in zip(fold_idx,time_gen[:,0]):
    plt.text(x - 0.225, y, '%.1f' % y, ha='center', va= 'bottom',fontsize = 8)
for x, y in zip(fold_idx, time_gen[:, 1]):
    plt.text(x + 0.225, y, '%.1f' % y, ha='center', va='bottom',fontsize = 8)


plt.figure()
plt.bar(left=fold_idx, height=error_age, width=0.8)
plt.xlabel('fold')
plt.ylabel('weighted error rate')
plt.title('error rate of prediction of age for different folds')
for x,y in zip(fold_idx,error_age):
    plt.text(x, y, '%.3f' % y, ha='center', va= 'bottom')

plt.figure()
plt.bar(left=fold_idx - 0.225, height=time_age[:,0], width=0.45, facecolor = 'lightskyblue')
plt.bar(left=fold_idx + 0.225, height= time_age[:,1], width=0.45,facecolor = 'yellowgreen')
plt.legend(['runtime of training', 'runtime of test'], loc= 'center left')
plt.xlabel('fold')
plt.ylabel('time (seconds)')
plt.title('runtime of training and test for prediction of age')
for x,y in zip(fold_idx,time_age[:,0]):
    plt.text(x - 0.225, y, '%.1f' % y, ha='center', va= 'bottom',fontsize = 8)
for x, y in zip(fold_idx, time_age[:, 1]):
    plt.text(x + 0.225, y, '%.1f' % y, ha='center', va='bottom',fontsize = 8)


plt.show()