#!/usr/bin/env python
# coding: utf-8

# # Genealogy
# 
# - Adding information for every reletative with a genetic overlapp ~ > 10% to DB
# - Adding the cancer set from cancer scripts 
# - Baseline information for each individual
# 

# In[3]:


# loading packages
import h5py
import sys
import pickle

import pandas as pd
import numpy as np 

from multiprocessing import Pool, TimeoutError


# In[4]:


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

def _write(name, p, sex, birthyear, status, EOO, cancer, degree):
    try: 
        del f[idx]['genealogy'][name]
    except:
        pass
    f[idx]['genealogy'].create_group(name)
    f[idx]['genealogy'][name].attrs['pid'] = p
    f[idx]['genealogy'][name].attrs['sex'] = sex
    f[idx]['genealogy'][name].attrs['birthyear'] = birthyear
    f[idx]['genealogy'][name].attrs['status'] = status
    f[idx]['genealogy'][name].attrs['EOO'] = EOO
    f[idx]['genealogy'][name].attrs['degree'] = degree
    f[idx]['genealogy'][name].create_dataset('cancer', data=cancer, maxshape=(None, 3), compression="lzf")


# In[29]:


####### grand_aunt/uncle    
def wide_family(pid_1, pid_2, pid_3, file_name):
        pid_siblings, pid_half_siblings_mother, pid_half_siblings_father = _siblings(pid_1, pid_2, pid_3)
        try :
            del f[idx]['genealogy/' +  file_name]['siblings']
            f[idx]['genealogy/' +  file_name].create_group('siblings')
        except:
            f[idx]['genealogy/' +  file_name].create_group('siblings')
            
        n = 0 
        for sib in pid_siblings:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.125)
            _write(file_name + '/siblings/_' + str(n), p, sex, birthyear, status, EOO, cancer, degree)
            f[idx]['genealogy/' + file_name + '/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.0625)
                _write(file_name + '/siblings/_' + str(n) + '/children/_' + str(nn), p, sex, birthyear, status, EOO, cancer, degree)
                f[idx]['genealogy/' + file_name + '/siblings/_' + str(n) + '/children/_' + str(nn)].create_group('children')
                pid_children2 = _children(kid)
                nnn = 0
                for kid2 in pid_children2:
                    p, sex, birthyear, status, EOO, cancer, degree = find_values(kid2, degree=0.03125)
                    _write(file_name + '/siblings/_' + str(n) + '/children/_' + str(nn) + '/children/_' + str(nnn), p, sex, birthyear, status, EOO, cancer, degree)
                    nnn += 1
                nn += 1
            n += 1
        for sib in pid_half_siblings_mother:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.0626)
            _write(file_name + '/siblings/_' + str(n), p, sex, birthyear, status, EOO, cancer, degree)
            f[idx]['genealogy/' + file_name + '/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.03126)
                _write(file_name + '/siblings/_' + str(n) + '/children/_' + str(nn), p, sex, birthyear, status, EOO, cancer, degree)
                f[idx]['genealogy/' + file_name + '/siblings/_' + str(n) + '/children/_' + str(nn)].create_group('children')
                pid_children2 = _children(kid)
                nnn = 0
                for kid2 in pid_children2:
                    p, sex, birthyear, status, EOO, cancer, degree = find_values(kid2, degree=0.015626)
                    _write(file_name + '/siblings/_' + str(n) + '/children/_' + str(nn) + '/children/_' + str(nnn), p, sex, birthyear, status, EOO, cancer, degree)
                    nnn += 1
                nn += 1
            n += 1
        for sib in pid_half_siblings_father:
            p, sex, birthyear, status, EOO, cancer, degree = find_values(sib, degree=0.0624)
            _write(file_name + '/siblings/_' + str(n), p, sex, birthyear, status, EOO, cancer, degree)
            f[idx]['genealogy/' + file_name + '/siblings/_' + str(n)].create_group('children')
            pid_children = _children(sib)
            nn = 0
            for kid in pid_children:
                p, sex, birthyear, status, EOO, cancer, degree = find_values(kid, degree=0.03124)
                _write(file_name + '/siblings/_' + str(n) + '/children/_' + str(nn), p, sex, birthyear, status, EOO, cancer, degree)
                f[idx]['genealogy/' + file_name + '/siblings/_' + str(n) + '/children/_' + str(nn)].create_group('children')
                pid_children2 = _children(kid)
                nnn = 0
                for kid2 in pid_children2:
                    p, sex, birthyear, status, EOO, cancer, degree = find_values(kid2, degree=0.015624)
                    _write(file_name + '/siblings/_' + str(n) + '/children/_' + str(nn) + '/children/_' + str(nnn), p, sex, birthyear, status, EOO, cancer, degree)
                    nnn += 1
                nn += 1
            n += 1


# In[6]:


# path
d_data = '/home/people/alexwolf/data/'
d_dnpr = '/home/projects/registries/2018/classic_style_lpr'


# In[7]:


# replace missing values
nrows = None
missing = {'0000000000': ''}


# In[8]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')


# In[9]:


# cluster
var1 = sys.argv[1]
var1 = int(var1)
print(var1)


# In[ ]:





# In[10]:


# load personal data
file = open(d_dnpr + '/preprocessing/prepared_data/t_person.tsv', 'r')
ancestry = pd.read_csv(file, sep='\t', nrows=None, usecols=[0, 6, 7,])
file.close()

ancestry.replace(missing, inplace=True)
ancestry = np.asarray(ancestry)
idx_list = ancestry[:, 0]


# In[11]:


# load cancer data 
# dictonary with cancer set for each pid
cancer_ref = pickle.load( open(d_data + 'DB/genealogy/_ref.pkl',"rb"))

# dictonary with baseline information for each pid 
personel_ref = pickle.load( open(d_data + 'DB/genealogy/_ref2.pkl',"rb"))


# In[30]:


with h5py.File(d_data + 'DB/DB/raw/_' + str(var1), 'a') as f:
        for idx in ref[var1]:
            ####### parents
            pid_mother, pid_father = _parents(idx)

            ####### grand_parents
            pid_mm, pid_mf = _parents(pid_mother)

            pid_fm, pid_ff = _parents(pid_father)

            ####### grand_grand_parents
            pid_mmm, pid_mmf = _parents(pid_mm)

            pid_mfm, pid_mff = _parents(pid_mf)

            pid_fmm, pid_fmf = _parents(pid_fm)

            pid_ffm, pid_fff = _parents(pid_ff)

            ####### grand_aunt/uncle

            wide_family(pid_mm, pid_mmm, pid_mmf, 'mother/mother')

            wide_family(pid_mf, pid_mfm, pid_mff, 'mother/father')

            wide_family(pid_fm, pid_fmm, pid_fmf, 'father/mother')

            wide_family(pid_ff, pid_ffm, pid_fff, 'father/father')



# In[21]:


print('finished')

