#!/usr/bin/env python
# coding: utf-8

# Extract the location of cancer individuals in the HDF5 files for sampling

# In[2]:


# loading packages
import sys
import os 
import h5py
import pickle
import torch 

import numpy as np 

import multiprocessing as mp

from itertools import chain
from multiprocessing import Pool

d_data = '/home/people/alexwolf/data/'

np.random.seed(7)

_cancer = 'C' + str(int(sys.argv[1]))


# In[3]:


# custom functions
def in_(x, y):
    a = []
    for ii in x: 
        a.append(ii in y)
    return(np.asarray(a))
    


# In[7]:


# sex specification
cancer_female_specific = ['C' + str(i) for i in np.arange(50, 59)]
cancer_male_specific = ['C' + str(i) for i in np.arange(60, 64)]

if _cancer in cancer_female_specific:
    sex_specific = 0
elif _cancer in cancer_male_specific:
    sex_specific = 1
else:
    sex_specific = None
    
    
# malignant cancer
malignant_cancer_codes = ['C' + str(i) for i in np.arange(10, 77)]
malignant_cancer_codes.extend(['C' + str(i) for i in np.arange(80, 100)])
malignant_cancer_codes.extend(['C0' + str(i) for i in np.arange(0, 10)])



# In[8]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy', allow_pickle=True)

# train-valid-test split 
data_split = pickle.load(open(d_data + 'DB/DB/raw/trainvalidtest.pickle', 'rb'))

# iterator lists
iterator_list_train = list(chain.from_iterable([list(enumerate(np.repeat(x, len(ref[x])), 0)) for x in data_split['train']]))
iterator_list_valid = list(chain.from_iterable([list(enumerate(np.repeat(x, len(ref[x])), 0)) for x in data_split['valid']]))


# In[ ]:


# extracting sample ids for proportional sampling
def find_cancer_ids(file, dir_=d_data + 'DB/DB/raw/_', ref=ref, cancer=_cancer):
    res = []
    with h5py.File(d_data + 'DB/DB/raw/_' + str(file), 'r') as f:
            n = 0
            for idx in ref[file]:
                if f[idx].attrs['cancer'] == 1:
                    cancer_set = f[idx]['cancer']['set'][:].astype(str)
                    idx_malignant = in_(cancer_set[:, 1], malignant_cancer_codes)
                    
                    if np.sum(idx_malignant) > 0:
                        primary = ((cancer_set[idx_malignant, 0].astype('datetime64[D]') - np.min(cancer_set[idx_malignant, 0].astype('datetime64[D]'))).astype(int) < 31*3)
                        date = np.min(cancer_set[idx_malignant, 0].astype('datetime64[D]'))
                        cancer_set = cancer_set[idx_malignant, :]
                        cancer_set = cancer_set[primary, :]
                        _cc1 = np.any(cancer_set[:, 1] == _cancer)
                        _cc2 = date < np.datetime64('2015-01-01')
                        _cc3 = date >= np.datetime64('1984-01-01')
                        if np.logical_and(_cc1, np.logical_and(_cc2, _cc3)):
                            res.append((n, file))
                n += 1
    return(res)

with Pool(processes=6) as pool:
    cancer_ids_train = [x for x in pool.imap_unordered(find_cancer_ids, data_split['train'])]
cancer_ids_train = [item for sublist in cancer_ids_train for item in sublist]

with Pool(processes=6) as pool:
    cancer_ids_valid = [x for x in pool.imap_unordered(find_cancer_ids, data_split['valid'])]
cancer_ids_valid = [item for sublist in cancer_ids_valid for item in sublist]    
    
cids_train = []
hids_train = []
for ii in range(len(iterator_list_train)): 
    if iterator_list_train[ii] in cancer_ids_train:
        cids_train.append(ii)
    else:
        hids_train.append(ii)
cids_train = np.asarray(cids_train)
hids_train = np.asarray(hids_train)


cids_valid = []
hids_valid = []
for ii in range(len(iterator_list_valid)): 
    if iterator_list_valid[ii] in cancer_ids_valid:
        cids_valid.append(ii)
    else:
        hids_valid.append(ii)
cids_valid = np.asarray(cids_valid)
hids_valid = np.asarray(hids_valid)


np.save(d_data + 'DB/DB/cancer_locations/' + str(_cancer) + '_ctrain.npy', cids_train)
np.save(d_data + 'DB/DB/cancer_locations/' + str(_cancer) + '_htrain.npy', hids_train)
np.save(d_data + 'DB/DB/cancer_locations/' + str(_cancer) + '_cvalid.npy', cids_valid)
np.save(d_data + 'DB/DB/cancer_locations/' + str(_cancer) + '_hvalid.npy', hids_valid)


# In[ ]:


print('finished')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




