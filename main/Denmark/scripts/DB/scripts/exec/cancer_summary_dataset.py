#!/usr/bin/env python
# coding: utf-8

# ## Cancer summary data set generation
# 
# - script to generate a numpy array containing information for all cancers in Denmark
# - comprises all cancer incidence pulled out by DB with primary information
# - split by train valid test
# 
# 
# ### File structure:
# 
# train - valid - test 
# 
# Array [:, 10]:   
# idx, age, sex, cancer, date, morph, primary, time_to_death, status, quality
# 
# 
# ### Output:
# 'DB/incidence/cancer_data.h5'
# 
# 
# 

# In[19]:


# loading packages
import sys
import os 
import h5py
import pickle
import torch 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import torch.optim as optim
import multiprocessing as mp

from itertools import chain
from torch.utils.data import DataLoader
from multiprocessing import Pool
from tqdm import tqdm

d_data = '/home/people/alexwolf/data/'

np.random.seed(7)


# In[20]:


# custom functions 
def in_(x, y):
    a = []
    for ii in x: 
        a.append(ii in y)
    return(np.asarray(a))
    


# In[21]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')

# malignant cancer
malignant_cancer_codes = ['C' + str(i) for i in np.arange(10, 77)]
malignant_cancer_codes.extend(['C' + str(i) for i in np.arange(80, 100)])
malignant_cancer_codes.extend(['C0' + str(i) for i in np.arange(0, 10)])

# train-valid-test split 
data_split = pickle.load(open(d_data + 'DB/DB/raw/trainvalidtest.pickle', 'rb'))


# In[22]:


def _extract(file, ref):
    res = np.zeros((1, 9))
    with h5py.File(d_data + 'DB/DB/raw/_' + str(file), 'r') as f:
        for idx in ref[file]:
            try:
                if f[idx].attrs['cancer'] == 1:
                    birthdate = f[idx].attrs['birthdate'].astype('datetime64')
                    EOO = f[idx].attrs['EOO'].astype('datetime64')
                    sex = f[idx].attrs['sex']
                    cancer_set = f[idx]['cancer']['set'][:, [0,1,4]].astype(str)
                    idx_malignant = in_(cancer_set[:, 1], malignant_cancer_codes)
                    if np.sum(idx_malignant) > 0:
                        primary = ((cancer_set[idx_malignant, 0].astype('datetime64[D]') - np.min(cancer_set[idx_malignant, 0].astype('datetime64[D]'))).astype(int) < 31*3).astype(int)[:, None]
                        cancer_set = cancer_set[idx_malignant, :]
                        age = np.round((cancer_set[:, 0].astype('datetime64') - birthdate).astype(int))[:, None]
                        TTO = np.round((EOO - cancer_set[:, 0].astype('datetime64')).astype(int))[:, None]
                        sex = np.repeat(f[idx].attrs['sex'], cancer_set.shape[0])[:, None]
                        status = np.repeat(f[idx].attrs['status'], cancer_set.shape[0])[:, None]
                        id_ = np.repeat(idx, cancer_set.shape[0])[:, None]
                        d = np.concatenate((id_, age, sex, cancer_set, primary, TTO, status), axis=1)
                        res = np.concatenate((res, d), axis=0)
            except:
                print('error: ', file, idx)
    res = res[1:, :]
    
    return(res)
        
    


# In[24]:


data_valid = np.concatenate([_extract(file, ref) for file in data_split['valid']])


# In[25]:


data_test = np.concatenate([_extract(file, ref) for file in data_split['test']])


# In[26]:


data_train = np.concatenate([_extract(file, ref) for file in data_split['train']])


# In[ ]:


with h5py.File(d_data + 'DB/incidence/cancer_data.h5', 'w') as f:
    f.create_dataset("train", data=data_train.astype('S45'), maxshape=(None, 10), compression="gzip", compression_opts=9)
    f.create_dataset("valid", data=data_valid.astype('S45'), maxshape=(None, 10), compression="gzip", compression_opts=9)
    f.create_dataset("test", data=data_test.astype('S45'), maxshape=(None, 10), compression="gzip", compression_opts=9)
            


# In[ ]:


print('finished')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




