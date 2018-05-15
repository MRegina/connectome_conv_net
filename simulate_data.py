# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:32:29 2017
Code to create simulated data for the convolutional network
@author: mregina
"""
import numpy as np

from six.moves import cPickle as pickle


#%% normalize tensors to have maximal absolute value of 1 and mean of 0
def normalize_tensor(data_tensor):
    data_tensor-=np.mean(data_tensor)
    data_tensor/=np.max(np.abs(data_tensor))
    return data_tensor

#%%
#create tensor from clean connectomes
def create_conn_tensor(base_connectome,diff_connectome,rep,numROI):
   
    base_tensor=np.tile(base_connectome,(rep,1,1))
    diff_tensor=np.tile(diff_connectome,(rep,1,1))
    
    data_tensor=np.zeros([2*rep,numROI,numROI,1])
    data_tensor[:rep,:,:,0]=base_tensor
    data_tensor[rep:,:,:,0]=diff_tensor
               
    data_tensor=normalize_tensor(data_tensor)
    
    label=np.zeros(2*rep)
    label[rep:]=1
    return data_tensor, label
     
     
#%%
#create noise-tensor
def create_noise_tensor(rep, numROI):
    noise=np.random.normal(0,1,[2*rep,numROI,numROI,1])
    noise=noise+np.transpose(noise,(0,2,1,3))
    noise=normalize_tensor(noise)
    return noise
#%%
#create an example dataset from the connectomes
c_base=np.load("corr_base_connectome.npy")
c_diff=np.load("corr_diff_connectome.npy")

d_base=np.load("dtw_base_connectome.npy")
d_diff=np.load("dtw_diff_connectome.npy")

rep=75
numROI=499

data_tensor1,label=create_conn_tensor(c_base,c_diff,rep,numROI)
data_tensor2,label=create_conn_tensor(d_base,d_diff,rep,numROI)
noise=create_noise_tensor(rep,numROI)


#save tensors and labels with a noise level of 5
pickle_file = 'tensors_5_noiselevel.pickle'

try:
  f = open(pickle_file, 'wb')
  save = {
    'data_tensor1': data_tensor1+5*noise,
    'data_tensor2': data_tensor2+5*noise,
    'label': label
    
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise
