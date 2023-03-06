#!/usr/bin/env python
# coding: utf-8

# # Adding DNPR data
# 
# ### Data:  
# 
# - 'pid', 'patient_type', 'date', 'private', 'hospital_id', 'war_id', 'icd10', 'diagnosis', 'map_quality'
# 
# 
# ### Adjustments:   
# 
# - private hospital indicator
# - map_quality set to 0 for nan values

# ### Importing Modules

# In[1]:


import sys
import os
import datetime
import h5py
import pandas as pd
import numpy as np 
import multiprocessing as mp
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib as mpl

from multiprocessing import Pool

np.random.seed(seed=83457)

d_data = '/home/people/alexwolf/data/'
d_DNPR = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/diag_adms_mapped.tsv'
d_CR = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_tumor.tsv'
d_DR1 = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_dodsaarsag_1.tsv'
d_DR2 = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_dodsaarsag_2.tsv'
d_personal = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_person.tsv'


# ### Loading Data (DNPR)

# In[3]:


# loading data files
DNPR = pd.read_csv(open(d_DNPR, 'r'), sep='\t', usecols=[0, 3, 4, 6, 7, 8, 9, 10, 13], nrows=None)
DNPR.columns = ['pid', 'patient_type', 'date', 'private', 'hospital_id', 'war_id', 'icd10', 'diagnosis', 'map_quality']
DNPR.head()


# In[6]:


ref = np.load(d_data + 'DB/DB/raw/ref.npy')


# ### Adjustments

# In[4]:


# indicator for private hospital 
DNPR['private'] = np.asarray(DNPR['private'] == 'PRIVATE').astype(int)

# indicator for mapping quality from icd7 - icd10 
DNPR['map_quality'].replace({np.nan: 0}, inplace=True)
DNPR['map_quality'] = np.asarray(DNPR['map_quality']).astype(int)


# ### Writing to files

# In[5]:


DNPR = DNPR.groupby(['pid', 'date', 'icd10', 'diagnosis']).first()

def _write(x):
    with h5py.File(d_data + 'DB/DB/raw/_' + str(x), 'a') as f:
        for ii in ref[x]:
            try:
                del f[ii]['DNPR']
            except:
                print('')
            try:
                helpvar = DNPR.loc[ii].reset_index()
                helpvar.sort_values(by='date', inplace=True)
            except:
                helpvar = np.repeat('', 8)[None, :]
            f[ii].create_dataset('DNPR', data=np.asarray(helpvar).astype('S10'), maxshape=(None, 8), compression="lzf")
    return(x)
    
def _test(x):
    with h5py.File(d_data + 'DB/DB/raw/_' + str(x), 'r') as f:
        for ii in ref[x]:
            f[ii]['DNPR'].shape[0]
    return(x)


# In[ ]:


n = 0
for x in np.arange(ref.__len__()):
    if n % 10 == 0:
        file = open('/home/people/alexwolf/projects/DB/logs/progress.txt', 'w') 
        file.write(str(n)) 
        file.close() 
    _write(x)


# In[ ]:


print('finished')


# In[ ]:


for x in np.arange(ref.__len__()):
    _test(x)


# In[ ]:


print('finished - tested')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




