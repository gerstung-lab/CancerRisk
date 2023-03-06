#!/usr/bin/env python
# coding: utf-8

# # Genealogy
# 
# - Adding information for every reletative with a genetic overlapp ~ > 10% to DB
# - Adding the cancer set from cancer scripts 
# - Baseline information for each individual
# 

# In[2]:


# loading packages
import h5py
import sys
import pickle
import pandas as pd
import numpy as np 

ind = int(sys.argv[1])


# In[3]:


# path
d_data = '/home/people/alexwolf/data/'
d_dnpr = '/home/projects/registries/2018/classic_style_lpr'


# In[4]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')


# In[40]:


for var1 in range(10 * ind, 10 * (ind + 1)):
    try:
        print(var1)
        with h5py.File(d_data + 'DB/DB/raw/_' + str(var1), 'a') as f:
            for idx in ref[var1]:
                print(idx)
                for i in range(len(f[idx]['genealogy']['father']['siblings'])):
                    try:
                        f[idx]['genealogy']['father']['siblings']['_' + str(i)]['children']
                    except:
                        f[idx]['genealogy']['father']['siblings']['_' + str(i)].create_group('children')
                for i in range(len(f[idx]['genealogy']['mother']['siblings'])):
                    try:
                        f[idx]['genealogy']['mother']['siblings']['_' + str(i)]['children']
                    except:
                        f[idx]['genealogy']['mother']['siblings']['_' + str(i)].create_group('children')
    except:
        print('error: ', var1)


# In[ ]:


print('finished')


# In[ ]:





# In[ ]:





# In[ ]:




