#!/usr/bin/env python
# coding: utf-8

# # Genealogy
# 
# - Adding information for every reletative with a genetic overlapp ~ > 10% to DB
# - Adding the cancer set from cancer scripts 
# - Baseline information for each individual
# 

# In[1]:





# In[ ]:


# loading packages
import h5py
import sys
import pickle

import pandas as pd
import numpy as np 

# custom functions
def _or(x, y):
    return np.logical_or(x, y)

def _and(x, y):
    return np.logical_and(x, y)

def _siblings(pid, pid_m, pid_f):
    if pid_m != '':
        same_mother = ancestry[:, 1] == pid_m
    else:
        same_mother = False

    if pid_f != '':
        same_father = ancestry[:, 2] == pid_f
    else:
        same_father = False
        
    excluding_self = ancestry[:, 0] != pid
        
    idx_s = np.where(_and(excluding_self, _and(same_father, same_mother)))[0]
    idx_sm = np.where(_and(excluding_self, _and(~same_father, same_mother)))[0]
    idx_sf = np.where(_and(excluding_self, _and(same_father, ~same_mother)))[0]
    
    pid_siblings = ancestry[idx_s, 0]
    pid_half_siblings_mother = ancestry[idx_sm, 0]
    pid_half_siblings_father = ancestry[idx_sf, 0]
    
    return((pid_siblings, pid_half_siblings_mother, pid_half_siblings_father))

def _parents(pid):
    if _and(_and(pid != '', len(pid) > 25), pid in personel_ref.keys()):
        idx = np.where(ancestry[:, 0] == pid)[0]
        # pid parents 
        pid_mother = ancestry[idx, 1][0]
        pid_father = ancestry[idx, 2][0]
    else:
        pid_mother = ''
        pid_father = ''
    return((pid_mother, pid_father))

def _children(pid):
    if _and(pid != '', len(pid) > 25):
        idx_c = np.where(_or(ancestry[:, 1] == pid, ancestry[:, 2] == pid))[0]
        pid_children = ancestry[idx_c, 0]
    else: 
        pid_children = np.zeros((1, 1))[[]]
    return(pid_children)
    
def find_values(pid, degree):
    if _and(_and(pid != '', len(pid) > 25), pid in personel_ref.keys()):
        sex = np.asarray(personel_ref[pid][0]).astype(int)[None]
        birthyear = np.asarray(personel_ref[pid][1]).astype('S10')[None]
        status = np.asarray(personel_ref[pid][2]).astype(int)[None]
        EOO = np.asarray(personel_ref[pid][3]).astype('S10')[None]    
        cancer = cancer_ref[pid][2]
        degree = np.asarray([degree])
    else:
        sex = np.zeros((1,))[[]].astype(int)
        birthyear = np.zeros((1,))[[]].astype('S10')
        status = np.zeros((1,))[[]].astype(int)
        EOO = np.zeros((1,))[[]].astype('S10')
        cancer = np.zeros((1,3))[[]].astype('S10')
        degree = np.zeros((1,))[[]] 
    return((np.asarray(pid).astype('S45')[None], sex, birthyear, status, EOO, cancer, degree))

def _write(name):
    f[idx]['genealogy'].create_group(name)
    f[idx]['genealogy'][name].attrs['pid'] = p
    f[idx]['genealogy'][name].attrs['sex'] = sex
    f[idx]['genealogy'][name].attrs['birthyear'] = birthyear
    f[idx]['genealogy'][name].attrs['status'] = status
    f[idx]['genealogy'][name].attrs['EOO'] = EOO
    f[idx]['genealogy'][name].attrs['degree'] = degree
    f[idx]['genealogy'][name].create_dataset('cancer', data=cancer, maxshape=(None, 3), compression="lzf")
    
# path
d_data = '/home/people/alexwolf/data/'
d_dnpr = '/home/projects/registries/2018/classic_style_lpr'

# replace missing values
nrows = None
missing = {'0000000000': ''}

# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')

# load personal data
file = open(d_dnpr + '/preprocessing/prepared_data/t_person.tsv', 'r')
ancestry = pd.read_csv(file, sep='\t', nrows=None, usecols=[0, 6, 7,])
file.close()

ancestry.replace(missing, inplace=True)
ancestry = np.asarray(ancestry)
idx_list = ancestry[:, 0]

# load cancer data 
# dictonary with cancer set for each pid
cancer_ref = pickle.load( open(d_data + 'DB/genealogy/_ref.pkl',"rb"))

# dictonary with baseline information for each pid 
personel_ref = pickle.load( open(d_data + 'DB/genealogy/_ref2.pkl',"rb"))

# cluster
counter = sys.argv[1]
counter = int(counter)

for var1 in np.arange(counter * 10, (counter + 1)*10):
    try:
        print(var1)
        with h5py.File(d_data + 'DB/DB/raw/_' + str(var1), 'a') as f:
            for idx in ref[var1]:
                # children
                pid_children = _children(idx)
                f[idx]['genealogy'].create_group('children')
                nn = 0
                for kid in pid_children:
                    p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.5)
                    _write('children/_' + str(nn))
                    nn += 1
    except:
        print('error: ', var1)
        


# In[2]:


print('finished')


# In[3]:




