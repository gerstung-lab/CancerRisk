#!/usr/bin/env python
# coding: utf-8

# In[27]:


# loading packages
import h5py
import sys
import pickle
import pandas as pd
import numpy as np 


# In[28]:


# path
d_data = '/home/people/alexwolf/data/'
d_dnpr = '/home/projects/registries/2018/classic_style_lpr'


# In[29]:


# load reference data 
ref = np.load(d_data + 'DB/DB/raw/ref.npy')


# In[30]:


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


# In[31]:


# functions 
# custom functions
def _or(x, y):
    return np.logical_or(x, y)

def _and(x, y):
    return np.logical_and(x, y)

def _ratio(a, b):
    if b == 0:
        return(str(np.nan))
    else:
        return(str(a/b))
    
def _match_birthyear(birthdate, sex, r1, r2, r3, r4):
    cond1 = res[:, 1] == birthdate
    cond2 = res[:, 3] == sex
    cond3 = res[:, -1] == 'nan'
    cond4 = _or(_and(res[:, 4] == 'nan', r1 == 'nan'), _and(res[:, 4] != 'nan', r1 != 'nan'))
    cond5 = _or(_and(res[:, 5] == 'nan', r2 == 'nan'), _and(res[:, 5] != 'nan', r2 != 'nan'))
    cond6 = _or(_and(res[:, 6] == 'nan', r3 == 'nan'), _and(res[:, 6] != 'nan', r3 != 'nan'))
    cond7 = _or(_and(res[:, 7] == 'nan', r4 == 'nan'), _and(res[:, 7] != 'nan', r4 != 'nan'))
    try:
        return(np.where(_and(cond7, _and(cond6, _and(cond5, _and(cond4, _and(cond1, _and(cond2, cond3)))))))[0][0])
    except:
        pass
    

def _extract(name):
    member = int(f[idx]['genealogy'][name].attrs['pid'] != b'')
    cset = f[idx]['genealogy'][name]['cancer'][:, 0].astype(str).tolist()
    cset = list(set(malignant_cancer_codes).intersection(cset))
    cancer = int(cset.__len__() >= 1)
    p = f[idx]['genealogy'][name].attrs['pid'].astype(str)[0]
    d = f[idx]['genealogy'][name].attrs['degree']
    return((p, d, member, cancer, cset))
    
def _genealogy(idx):
    d1_members = 0 
    d1_cancer_cases = 0
    d1_cancer_set = []

    d2_members = 0 
    d2_cancer_cases = 0
    d2_cancer_set = []

    d3_members = 0 
    d3_cancer_cases = 0
    d3_cancer_set = []

    pid_set = []

    # father 
    p, d, member, cancer, cset =  _extract('father')
    d1_members += member
    d1_cancer_cases += cancer
    d1_cancer_set.append(cset)
    pid_set.append(p)

    # mother 
    p, d, member, cancer, cset =  _extract('mother')
    d1_members += member
    d1_cancer_cases += cancer
    d1_cancer_set.append(cset)
    pid_set.append(p)

    # siblings - half siblings
    for i in np.arange(f[idx]['genealogy']['siblings'].__len__()):
        p, d, member, cancer, cset =  _extract('siblings/_' + str(i))
        if d > 0.4:
            d1_members += member
            d1_cancer_cases += cancer
            d1_cancer_set.append(cset)
            pid_set.append(p)
        else:
            d2_members += member
            d2_cancer_cases += cancer
            d2_cancer_set.append(cset)
            pid_set.append(p)

    # aunt/uncle - cousin        
    for i in np.arange(f[idx]['genealogy']['father']['siblings'].__len__()):
        p, d, member, cancer, cset =  _extract('father/siblings/_' + str(i))
        if d > 0.2:
            d2_members += member
            d2_cancer_cases += cancer
            d2_cancer_set.append(cset)
            pid_set.append(p)
        else:
            d3_members += member
            d3_cancer_cases += cancer
            d3_cancer_set.append(cset)
            pid_set.append(p)
        for j in np.arange(f[idx]['genealogy']['father']['siblings']['_' + str(i)]['children'].__len__()): 
            p, d, member, cancer, cset =  _extract('father/siblings/_' + str(i) + '/children/_' + str(j))
            d3_members += member
            d3_cancer_cases += cancer
            d3_cancer_set.append(cset)
            pid_set.append(p)

    # grandparents - grand uncle/aunt 
    p, d, member, cancer, cset =  _extract('mother/mother')
    d2_members += member
    d2_cancer_cases += cancer
    d2_cancer_set.append(cset)
    pid_set.append(p)
    for i in np.arange(f[idx]['genealogy']['mother/mother/siblings'].__len__()):
        p, d, member, cancer, cset =  _extract('mother/mother/siblings/_' + str(i))
        d3_members += member
        d3_cancer_cases += cancer
        d3_cancer_set.append(cset)
        pid_set.append(p)

    p, d, member, cancer, cset =  _extract('mother/father')
    d2_members += member
    d2_cancer_cases += cancer
    d2_cancer_set.append(cset)
    pid_set.append(p)
    for i in np.arange(f[idx]['genealogy']['mother/father/siblings'].__len__()):
        p, d, member, cancer, cset =  _extract('mother/father/siblings/_' + str(i))
        d3_members += member
        d3_cancer_cases += cancer
        d3_cancer_set.append(cset)
        pid_set.append(p)

    p, d, member, cancer, cset =  _extract('father/mother')
    d2_members += member
    d2_cancer_cases += cancer
    d2_cancer_set.append(cset)
    pid_set.append(p)
    for i in np.arange(f[idx]['genealogy']['father/mother/siblings'].__len__()):
        p, d, member, cancer, cset =  _extract('father/mother/siblings/_' + str(i))
        d3_members += member
        d3_cancer_cases += cancer
        d3_cancer_set.append(cset)
        pid_set.append(p)

    p, d, member, cancer, cset =  _extract('father/father')
    d2_members += member
    d2_cancer_cases += cancer
    d2_cancer_set.append(cset)
    pid_set.append(p)
    for i in np.arange(f[idx]['genealogy']['father/father/siblings'].__len__()):
        p, d, member, cancer, cset =  _extract('father/father/siblings/_' + str(i))
        d3_members += member
        d3_cancer_cases += cancer
        d3_cancer_set.append(cset)
        pid_set.append(p)

    # grand grand parents
    p, d, member, cancer, cset =  _extract('mother/mother/mother')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('mother/mother/father')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('mother/father/mother')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('mother/father/father')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('father/mother/mother')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('father/mother/father')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('father/father/mother')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)

    p, d, member, cancer, cset =  _extract('father/father/father')
    d3_members += member
    d3_cancer_cases += cancer
    d3_cancer_set.append(cset)
    pid_set.append(p)
    pid_set = set(pid_set)
    try: 
        pid_set.remove('')
    except:
        pass
    return(d1_members, d1_cancer_cases, d1_cancer_set, d2_members, d2_cancer_cases, d2_cancer_set, d3_members, d3_cancer_cases, d3_cancer_set, pid_set)



# In[34]:


pid_set = set()
for var1 in np.arange(0, 750):
    res = []
    try:
        with h5py.File(d_data + 'DB/DB/raw/_' + str(var1), 'r') as f:
            for idx in ref[var1]:
                if f[idx].attrs['cancer'] == 1:
                    if f[idx]['genealogy']['mother'].attrs['pid'] != b'':
                        d1_members, d1_cancer_cases, d1_cancer_set, d2_members, d2_cancer_cases, d2_cancer_set, d3_members, d3_cancer_cases, d3_cancer_set, p = _genealogy(idx)
            
                        
                        if pid_set.intersection(p).__len__() == 0:
                            d1_cancer_same = np.sum([f[idx]['cancer'].attrs['icd10'].astype(str)[0] in x for x in d1_cancer_set])
                            d2_cancer_same = np.sum([f[idx]['cancer'].attrs['icd10'].astype(str)[0] in x for x in d2_cancer_set])
                            d3_cancer_same = np.sum([f[idx]['cancer'].attrs['icd10'].astype(str)[0] in x for x in d3_cancer_set])
                            d_cancer_same = d1_cancer_same + d2_cancer_same + d3_cancer_same
                            _r = []
                            _r.append(f[idx]['cancer'].attrs['icd10'].astype(str)[0])
                            _r.append(f[idx].attrs['birthdate'].astype(str)[0][:4])
                            _r.append(np.round((f[idx]['cancer'].attrs['date'].astype('datetime64') - f[idx].attrs['birthdate'].astype('datetime64')).astype(int)[0] / 365).astype(int))
                            _r.append(f[idx].attrs['sex'][0])
                            _r.append(_ratio(d1_cancer_cases, d1_members))
                            _r.append(_ratio(d2_cancer_cases, d2_members))
                            _r.append(_ratio(d3_cancer_cases, d3_members))
                            _r.append(_ratio(d1_cancer_cases + d2_cancer_cases + d3_cancer_cases, d1_members + d2_members + d3_members))
                            
                            _r.append(_ratio(d1_cancer_same, d1_members))
                            _r.append(_ratio(d2_cancer_same, d2_members))
                            _r.append(_ratio(d3_cancer_same, d3_members))
                            _r.append(_ratio(d_cancer_same, d1_members + d2_members + d3_members))

                            pid_set.update(p)
                            res.append(_r)
        res = np.asarray(res)
        res = np.concatenate((res, np.zeros((res.shape[0], 9)) * np.nan), axis=1)

        with h5py.File(d_data + 'DB/DB/raw/_' + str(var1), 'r') as f:
            for idx in ref[var1]:
                if f[idx].attrs['cancer'] == 0:
                    if f[idx]['genealogy']['mother'].attrs['pid'] != b'':
                        d1_members, d1_cancer_cases, d1_cancer_set, d2_members, d2_cancer_cases, d2_cancer_set, d3_members, d3_cancer_cases, d3_cancer_set, p = _genealogy(idx)
                        if pid_set.intersection(p).__len__() == 0:
                            birthdate = f[idx].attrs['birthdate'].astype(str)[0][:4]
                            sex = f[idx].attrs['sex'][0].astype(str)
                            r1 = _ratio(d1_cancer_cases, d1_members)
                            r2 = _ratio(d2_cancer_cases, d2_members)
                            r3 = _ratio(d3_cancer_cases, d3_members)
                            r4 = _ratio(d1_cancer_cases + d2_cancer_cases + d3_cancer_cases, d1_members + d2_members + d3_members)
                            identifyer = _match_birthyear(birthdate, sex, r1, r2, r3, r4)
                            if identifyer != None:                          
                                d1_cancer_same = np.sum([res[identifyer, 0] in x for x in d1_cancer_set])
                                d2_cancer_same = np.sum([res[identifyer, 0] in x for x in d2_cancer_set])
                                d3_cancer_same = np.sum([res[identifyer, 0] in x for x in d3_cancer_set])
                                d_cancer_same = d1_cancer_same + d2_cancer_same + d3_cancer_same
                                r5 = _ratio(d1_cancer_same, d1_members)
                                r6 = _ratio(d2_cancer_same, d2_members)
                                r7 = _ratio(d3_cancer_same, d3_members)
                                r8 = _ratio(d_cancer_same, d1_members + d2_members + d3_members)
                                res[identifyer, 12] = r1
                                res[identifyer, 13] = r2
                                res[identifyer, 14] = r3
                                res[identifyer, 15] = r4    
                                res[identifyer, 16] = r5
                                res[identifyer, 17] = r6
                                res[identifyer, 18] = r7
                                res[identifyer, 19] = r8
                                res[identifyer, 20] = '1'
                                pid_set.update(p)
                            else:
                                pass
        with h5py.File(d_data + 'DB/genealogy/comparision2', 'a') as f:
            f.create_dataset('data_' + str(var1), data=res.astype('S10'), maxshape=(None, 21), compression="lzf")     
        file1 = open(d_data + 'DB/genealogy/progress2.txt',"w")
        file1.write(str(var1)) 
        file1.close() 
    except:
        pass


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




