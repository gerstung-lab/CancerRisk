#!/usr/bin/env python
# coding: utf-8

# # Genealogy
# 
# - Adding information for every reletative with a genetic overlapp ~ > 10% to DB
# - Adding the cancer set from cancer scripts 
# - Baseline information for each individual
# 

# In[1]:


# loading packages
import h5py
import sys
import pickle

import pandas as pd
import numpy as np 


# In[2]:


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
    


# In[3]:


# path
d_data = '/home/people/alexwolf/data/'
d_dnpr = '/home/projects/registries/2018/classic_style_lpr'


# In[4]:


# replace missing values
nrows = None
missing = {'0000000000': ''}


# In[5]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')


# In[ ]:


# cluster
var1 = sys.argv[1]
var1 = int(var1)
print(var1)


# In[6]:


# load personal data
file = open(d_dnpr + '/preprocessing/prepared_data/t_person.tsv', 'r')
ancestry = pd.read_csv(file, sep='\t', nrows=None, usecols=[0, 6, 7,])
file.close()

ancestry.replace(missing, inplace=True)
ancestry = np.asarray(ancestry)
idx_list = ancestry[:, 0]


# In[7]:


# load cancer data 
# dictonary with cancer set for each pid
cancer_ref = pickle.load( open(d_data + 'DB/genealogy/_ref.pkl',"rb"))

# dictonary with baseline information for each pid 
personel_ref = pickle.load( open(d_data + 'DB/genealogy/_ref2.pkl',"rb"))


# In[8]:


with h5py.File(d_data + 'DB/DB/raw/_' + str(var1), 'a') as f:
    for idx in ref[var1]:
        try:
            del f[idx]['genealogy']
            f[idx].create_group('genealogy')
        except:
            f[idx].create_group('genealogy')

        ####### parents
        pid_mother, pid_father = _parents(idx)
        # mother
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mother, degree=0.5)
        _write('mother')

        # father
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_father, degree=0.5)
        _write('father')
        
        # children
        pid_children = _children(idx)
        f[idx]['genealogy'].create_group('children')
        nn = 0
        for kid in pid_children:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.5)
            _write('children/_' + str(nn))
            nn += 1

        ####### grand_parents
        pid_mm, pid_mf = _parents(pid_mother)
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mm, degree=0.25)
        _write('mother/mother')
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mf, degree=0.25)
        _write('mother/father')


        pid_fm, pid_ff = _parents(pid_father)
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_fm, degree=0.25)
        _write('father/mother')
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_ff, degree=0.25)
        _write('father/father')


        ####### grand_grand_parents
        pid_mmm, pid_mmf = _parents(pid_mm)
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mmm, degree=0.125)
        _write('mother/mother/mother')
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mmf, degree=0.125)
        _write('mother/mother/father')

        pid_mfm, pid_mff = _parents(pid_mf)
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mfm, degree=0.125)
        _write('mother/father/mother')
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_mff, degree=0.125)
        _write('mother/father/father')

        pid_fmm, pid_fmf = _parents(pid_fm)
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_fmm, degree=0.125)
        _write('father/mother/mother')
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_fmf, degree=0.125)
        _write('father/mother/father')

        pid_ffm, pid_fff = _parents(pid_ff)
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_ffm, degree=0.125)
        _write('father/father/mother')
        p, sex, birthyear, status, EOO, cancer, degree = find_values(pid_fff, degree=0.125)
        _write('father/father/father')

        ####### grand_aunt/uncle
        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_mm, pid_mmm, pid_mmf)
        f[idx]['genealogy/mother/mother'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.125)
            _write('mother/mother/siblings/_' + str(n))
            n += 1

        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_mf, pid_mfm, pid_mff)
        f[idx]['genealogy/mother/father'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.125)
            _write('mother/father/siblings/_' + str(n))
            n += 1

        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_fm, pid_fmm, pid_fmf)
        f[idx]['genealogy/father/mother'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.125)
            _write('father/mother/siblings/_' + str(n))
            n += 1

        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_ff, pid_ffm, pid_fff)
        f[idx]['genealogy/father/father'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.125)
            _write('father/father/siblings/_' + str(n))
            n += 1

        #######  aunt uncle + cousins
        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_mother, pid_mm, pid_mf)
        f[idx]['genealogy/mother'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.25)
            _write('mother/siblings/_' + str(n))
            f[idx]['genealogy/mother/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.125)
                _write('mother/siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1
        for sib in pid_half_siblings_mother:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.126)
            _write('mother/siblings/_' + str(n))
            f[idx]['genealogy/mother/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.0626)
                _write('mother/siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1   
        for sib in pid_half_siblings_father:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.124)
            _write('mother/siblings/_' + str(n))
            f[idx]['genealogy/mother/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.0624)
                _write('mother/siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1  

        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_father, pid_fm, pid_ff)
        f[idx]['genealogy/father'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.25)
            _write('father/siblings/_' + str(n))
            f[idx]['genealogy/father/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.125)
                _write('father/siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1
        for sib in pid_half_siblings_mother:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.126)
            _write('father/siblings/_' + str(n))
            f[idx]['genealogy/father/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.0626)
                _write('father/siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1   
        for sib in pid_half_siblings_father:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.124)
            _write('father/siblings/_' + str(n))
            f[idx]['genealogy/father/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.0624)
                _write('father/siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1  

        ####### siblings
        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(idx, pid_mother, pid_father)
        f[idx]['genealogy'].create_group('siblings')
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.5)
            _write('siblings/_' + str(n))
            f[idx]['genealogy/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.25)
                _write('siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1
        for sib in pid_half_siblings_mother:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.26)
            _write('siblings/_' + str(n))
            f[idx]['genealogy/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.126)
                _write('siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1   
        for sib in pid_half_siblings_father:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.24)
            _write('siblings/_' + str(n))
            f[idx]['genealogy/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.124)
                _write('siblings/_' + str(n) + '/children/_' + str(nn))
                nn += 1
            n += 1  




# In[ ]:


print('finished')


# In[ ]:





# In[ ]:





# In[ ]:




