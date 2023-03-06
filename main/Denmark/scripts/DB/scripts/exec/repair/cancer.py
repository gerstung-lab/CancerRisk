#!/usr/bin/env python
# coding: utf-8

# # Cancer information for DB
# 
# - script to add the corresponding cancer information to the DB 
# - Combination of information from DNPR - CR - DR 
# - DR codes befor 1994 are transformed using icd8_icd10_cancer_map.txt
# 
# ### Output:
# 
# - cancer group to HDF5 files in DB  
# 
# Datasets:  
# DNPR - CR - DR - DR_full - set - set_full
# 
# Covar:
# 'date', 'icd10', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr'
# 

# In[9]:


import sys
import os
import datetime
import h5py
import glob
import pickle
import pandas as pd
import numpy as np 
import multiprocessing as mp
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib as mpl

from multiprocessing import Pool

np.random.seed(seed=83457)

d_data = '/home/people/alexwolf/data/'

# refernce files
ref_DB = np.load(d_data + 'DB/DB/raw/ref.npy')
ref_cancer = pickle.load(open(d_data + 'DB/cancer/_ref.pickle', "rb"))


# Cancer Definition:

# In[10]:


# defining lists for diagnosisi
# uncertain cancer diagnosis 
uncertain_cancer_codes = ['D' + str(i) for i in np.arange(37, 45)]
uncertain_cancer_codes.extend(['D' + str(i) for i in np.arange(47, 49)])

# bening cancer
benign_cancer_codes = ['D' + str(i) for i in np.arange(10, 37)]
benign_cancer_codes.extend(['D0' + str(i) for i in np.arange(0, 10)])
benign_cancer_codes.extend(['D45', 'D46'])

# malignant cancer
malignant_cancer_codes = ['C' + str(i) for i in np.arange(10, 77)]
malignant_cancer_codes.extend(['C' + str(i) for i in np.arange(80, 100)])
malignant_cancer_codes.extend(['C0' + str(i) for i in np.arange(0, 10)])

# treatment for malignant cancer
malignant_cancer_treatment_codes = ['Z08', 'Z85']

# secondary cancers 
secondary_cancer_codes = ['C77', 'C78', 'C79']

relevant_diagnosis = secondary_cancer_codes + malignant_cancer_treatment_codes + malignant_cancer_codes + uncertain_cancer_codes
cancer_diagnosis = malignant_cancer_codes + secondary_cancer_codes


icd8_cancer = np.arange(140, 241).astype(str).tolist()


# Varibale passed from job submission

# In[7]:


# varibale passed from job submission
ind = int(sys.argv[1])


# In[ ]:





# Custom Functions:

# In[12]:


def _find(idx, ref):
    k = 9999
    for key, value in ref.items():
        if idx in value: 
            k = key
            break 
    return(k)

def _or(x, y):
    return np.logical_or(x, y)

def _and(x, y):
    return np.logical_and(x, y)

def _minimum(x, y):
    return np.minimum(x, y)

def _maximum(x, y):
    return np.maximum(x, y)

def remove_D_coding(x):
    ll = []
    for i in x:
        ll.append(i.replace('DC', 'C').replace('DD', 'D').replace('DZ', 'Z'))
    return(np.asarray(ll))

def _adjust_dr(x):
    if x.shape[0]>0:
        helpvar = []
        for jj in x[:, 1:].tolist()[0]:
            helpvar.extend(jj.replace(' ', '').split(','))
        helpvar = np.asarray(helpvar)[None, :]
        helpvar = np.unique(helpvar)
        helpvar = helpvar[helpvar != 'nan']
        helpvar = helpvar[helpvar != '']
        helpvar = helpvar[helpvar != ' ']
        helpvar = helpvar[helpvar != 'na']
        helpvar = helpvar[helpvar != ',']
        x = np.concatenate((np.asarray(x[0, 0])[None, None], helpvar[None, :]), axis=1)[0]
        x = np.concatenate([x[[0, j]][None, :] for j in range(1, x.shape[0])], axis=0)
        return(x)
    else:
        return(np.zeros((1, 2))[[]])


# In[13]:


for idx in ref_DB[ind]:
    file_id = _find(idx, ref_cancer)
    if file_id == 9999:
        DNPR = np.zeros((1,7))[[]]
        CR = np.zeros((1,5))[[]]
        DR = np.zeros((1,14))[[]]
        DR_full = np.zeros((1,14))[[]]
        cancer_set = np.zeros((1,8))[[]]
        cancer_set_full = np.zeros((1,8))[[]]
        cancer_idx = np.asarray([0]).astype(int)
    else:
        cancer_idx = np.asarray([1]).astype(int)
        with h5py.File(d_data + 'DB/cancer/_' + str(file_id) + '.h5', 'r') as f:
            # cancer information from DNPR
            dnpr = f[idx]['DNPR'][:].astype(str)
            if dnpr.shape[0] != 0:
                dnpr[:, -2] = remove_D_coding(dnpr[:, -2])
            DNPR = dnpr
            dnpr = dnpr[:, [0, 1, 5, 6]]
            dnpr = pd.DataFrame(dnpr) 
            dnpr.columns=['date', 'icd10', 'icd10_full', 'count']
            dnpr['idx_dnpr'] = 1

            # cancer information from CR
            cr = f[idx]['CR'][:].astype(str)
            cr[cr == 'nan'] = '0'
            CR = cr
            cr = pd.DataFrame(cr)
            cr.columns = ['date', 'tumour_count', 'icd10_full', 'morph', 'icd10']
            cr['idx_cr'] = 1

            # cancer information from DR
            dr = _adjust_dr(f[idx]['DR'][:, :14].astype(str))
            dr_full = _adjust_dr(np.concatenate((f[idx]['DR'][:, 0].astype(str)[:, None], f[idx]['DR'][:, 14:].astype(str)), axis=1))
            DR = dr
            DR_full = dr_full
            dr = pd.DataFrame(dr)
            dr.columns = ['date', 'icd10']
            dr['idx_dr'] = 1
            dr_full = pd.DataFrame(dr_full)
            dr_full.columns = ['date', 'icd10_full']
            dr_full['idx_dr'] = 1
            # remove non cancer icd8 codes 
            idx_keep = np.asarray(_or(_or(dr_full['icd10_full'].apply(lambda x: x[0]=='C'), dr_full['icd10_full'].apply(lambda x: x[0]=='D')), dr_full['icd10_full'].apply(lambda x: x[:3] in icd8_cancer)))
            dr_full = dr_full.loc[idx_keep, :]

            # combining cancer information across registries
            cancer_set = pd.DataFrame(columns=['date', 'icd10', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr'])
            cancer_set_full = pd.DataFrame(columns=['date', 'icd10_full', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr'])

            # Defining cancer set
            cancer_set = cancer_set.append(dnpr, sort=False)
            cancer_set = cancer_set.append(cr, sort=False)
            cancer_set = cancer_set.append(dr, sort=False)
            cancer_set = cancer_set.loc[:, ['date', 'icd10', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr']]
            cancer_set = cancer_set.fillna(0)
            cancer_set.loc[:, 'date'] = np.asarray(cancer_set.loc[:, 'date']).astype('datetime64')
            cancer_set.loc[:, 'count'] = np.asarray(cancer_set.loc[:, 'count']).astype(int)
            cancer_set.loc[:, 'tumour_count'] = np.asarray(cancer_set.loc[:, 'tumour_count']).astype(int)
            cancer_set.loc[:, 'morph'] = np.asarray(cancer_set.loc[:, 'morph']).astype(int)


            cancer_set_full = cancer_set_full.append(dnpr, sort=False)
            cancer_set_full = cancer_set_full.append(cr, sort=False)
            cancer_set_full = cancer_set_full.append(dr_full, sort=False)
            cancer_set_full = cancer_set_full.loc[:, ['date', 'icd10_full', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr']]
            cancer_set_full = cancer_set_full.fillna(0)
            cancer_set_full.loc[:, 'date'] = np.asarray(cancer_set_full.loc[:, 'date']).astype('datetime64')
            cancer_set_full.loc[:, 'count'] = np.asarray(cancer_set_full.loc[:, 'count']).astype(int)
            cancer_set_full.loc[:, 'tumour_count'] = np.asarray(cancer_set_full.loc[:, 'tumour_count']).astype(int)
            cancer_set_full.loc[:, 'morph'] = np.asarray(cancer_set_full.loc[:, 'morph']).astype(int)


            cancer_set = cancer_set.groupby('icd10').agg({'date': min, 
                                            'count': max, 
                                            'tumour_count': max, 
                                            'morph': max, 
                                            'idx_dnpr': max, 
                                            'idx_cr': max, 
                                            'idx_dr': max}).reset_index()

            cancer_set = cancer_set.sort_values('date')  
            cancer_set = cancer_set.loc[:, ['date', 'icd10', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr']]
            cancer_set.loc[:, 'date'] = cancer_set.loc[:, 'date'].astype(str)

            cancer_set_full = cancer_set_full.groupby('icd10_full').agg({'date': min, 
                                            'count': max, 
                                            'tumour_count': max, 
                                            'morph': max, 
                                            'idx_dnpr': max, 
                                            'idx_cr': max, 
                                            'idx_dr': max}).reset_index()

            cancer_set_full = cancer_set_full.sort_values('date')  
            cancer_set_full = cancer_set_full.loc[:, ['date', 'icd10_full', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr']]
            cancer_set_full.loc[:, 'date'] = cancer_set_full.loc[:, 'date'].astype(str)
            
    with h5py.File(d_data + 'DB/DB/raw/_' + str(ind), 'a') as f:
            try: 
                del f[idx]['cancer']
                del f[idx].attrs['cancer']
                f[idx].create_group('cancer')
                f[idx].attrs['cancer'] = cancer_idx
            except:
                f[idx].create_group('cancer')
                f[idx].attrs['cancer'] = cancer_idx

            f[idx]['cancer'].create_dataset('DNPR', data=DNPR.astype('S10'), maxshape=(None, 7), compression="lzf")
            f[idx]['cancer'].create_dataset('CR', data=CR.astype('S10'), maxshape=(None, 5), compression="lzf")
            f[idx]['cancer'].create_dataset('DR', data=DR.astype('S10'), maxshape=(None, 14), compression="lzf")
            f[idx]['cancer'].create_dataset('DR_full', data=DR_full.astype('S10'), maxshape=(None, 14), compression="lzf")
            f[idx]['cancer'].create_dataset('set', data=np.asarray(cancer_set).astype('S10'), maxshape=(None, 8), compression="lzf")
            f[idx]['cancer'].create_dataset('set_full', data=np.asarray(cancer_set_full).astype('S10'), maxshape=(None, 8), compression="lzf")

            


# In[ ]:


print('finsihed')

