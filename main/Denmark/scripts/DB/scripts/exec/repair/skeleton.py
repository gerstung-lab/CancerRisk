#!/usr/bin/env python
# coding: utf-8

# # Skeleton for Danish Database 
# 
# - script to generate the skeleton for the DB 
# - 10000 PIDs per HDF5 file
# - group level is PID - attributes (birthdate, sex, status, EOO (End of Observation))  
# 
# ### Adjustments:   
# 
# - EOO is either their last status date (dead - left country ect.)  or last obs (2018-04-10) 
# - Sex - Male = 1, Female = 0 
# 
# ### Output:   
# 
# - 1000 HDF5 files 
# - PIDs with their baseline information   
# 
# 'DB/DB/raw/_x'

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


# ### Loading CPR data - basis for skeleton 

# In[2]:


PD = pd.read_csv(open(d_personal, 'r'), sep='\t', nrows=None, usecols=[0,1,2,3,4,8])
PD.columns = ['pid', 'sex', 'birthdate', 'status', 'status_date', 'EOO']
PD = PD.sample(frac=1).reset_index(drop=True) # shuffel data
PD.head()



# ### Adjustments

# In[3]:


# replacing EOO with status
PD.loc[~np.asarray(PD['status_date'].isnull()), 'EOO'] = PD.loc[~np.asarray(PD['status_date'].isnull()), 'status_date']

# replace sex values
PD['sex'].replace({'K': 0, 
                   'M': 1}, inplace=True)


# 
# ### Writing to files

# In[4]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')
PD.set_index('pid', inplace=True, verify_integrity=True)

def _write(x):
    with h5py.File(d_data + 'DB/DB/raw/_' + str(x),'a') as f:
        for ii in ref[x]:
            f.create_group(str(ii))
            helpvar = PD.loc[ii]
            f[str(ii)].attrs['birthdate'] = np.asarray(helpvar['birthdate'])[None].astype('S10')
            f[str(ii)].attrs['sex'] = np.asarray(helpvar['sex'])[None].astype(int)
            f[str(ii)].attrs['status'] = np.asarray(helpvar['status'])[None].astype(int)
            f[str(ii)].attrs['EOO'] = np.asarray(helpvar['EOO'])[None].astype('S10')
    return(x)

def _test(x):
    with h5py.File(d_data + 'DB/DB/raw/_' + str(x), 'r') as f:
        for ii in ref[x]:
            f[str(ii)]
            f[str(ii)].attrs['birthdate']
    return(x)
    


# In[ ]:


# cluster
var1 = sys.argv[1]
var1 = int(var1)
_write(var1)


# In[ ]:





# In[ ]:





# In[ ]:




