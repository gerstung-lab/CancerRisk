# %% [markdown]
# # Cancer incidence for registries in Denmark (DNPR, CR, DR)
# Collecting all cancer information on an individual level about cancer incidence
# 
# ## Registries:
# - The Danish National Patient Registry (DNPR).
# - The Cancer regsitry (CR).
# - The Death Certificates Register (DR).   
# 
# ## Output:
# HDF5 files with group -level CPR and corresponding DNPR, CR, DR, cancer information.
# 
# ### Adjustments: 
# 
# #### Global:
# - icd10 - level 3 as basis  
# - only relevant diagnosis - C*, D37-45, Z08, Z85
# 
# #### DNPR: 
# - only primary and secondary diagnosis
# - collapse information on first entry
# - dummy for public hospital 
# 
# - var: 'pid', 'date', 'icd10', 'public', 'hospital', 'ward', 'code', 'count'
# 
# #### CR:
# - var: 'pid', 'date', 'tumour_count', 'code', 'morph', 'icd10'
# 
# #### PD:
# - var: 'pid', 'sex', 'birthdate', 'status', 'status_date'
# 
# #### DR:
# - transofrming DR1 cancer codes from icd8 to icd10
# 
# - var: 'pid', 'date', 'icd10_1', 'icd10_2', 'icd10_3', 'icd10_4', 'icd10_5', 'icd10_6', 'icd10_7', 'icd10_8', 'icd10_9', 'icd10_10', 'icd10_11', 'icd10_12', 'icd10_13', 'code1', 'code2','code3', 'code4', 'code5', 'code6', 'code7', 'code8','code9', 'code10', 'code11', 'code12', 'code13'
#        
# 
# 
# 
# 'DB/cancer/_' + str(ind)
# 

# %% [markdown]
#  

# %% [markdown]
# **Importing Moules:**

# %%
import sys
import os
import datetime
import h5py
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
d_DNPR = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/diag_adms_mapped.tsv'
d_CR = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_tumor.tsv'
d_DR1 = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_dodsaarsag_1.tsv'
d_DR2 = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_dodsaarsag_2.tsv'
d_personal = '/home/projects/registries/2018/classic_style_lpr/preprocessing/prepared_data/t_person.tsv'

# %% [markdown]
# **Cluster:**

# %%
# varibale passed from job submission
ind = int(sys.argv[1])

# %% [markdown]
# **Custom Functions:**

# %%
def adjust_data(data, relevant_diagnosis):
    '''
    transform diagnosis to 3rd-level icd10 international.
    extracts all relevant cancer diagnosis based in the list in relevant_diagnosis.
    only keep primary and secodnary diagnosis.
    '''
    # set datetime variable 
    data['icd10'] = data['CODE'].apply(lambda x: x[1:4])
    data = data.loc[data['icd10'].apply(lambda x: x in relevant_diagnosis)].reset_index(drop=True)
    data = data.loc[data['CODE_TYPE'].apply(lambda x: 'A' in x or 'B' in x)].reset_index(drop=True)
    data.drop('CODE_TYPE', axis=1, inplace=True)
    return(data)


# %% [markdown]
# **Data Preperation:**
# 

# %% [markdown]
# **ICD10 Codes**:   
# 

# %%
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


# %% [markdown]
# #### ICD8 to ICD10 cancer mapping:

# %%
mapping = pickle.load(open(d_data + 'DB/cancer/icd8_icd10_cancer_mapping.pickle', 'rb'))

# %% [markdown]
# **Personal Information:**

# %%
PD = pd.read_csv(open(d_personal, 'r'), sep='\t', usecols=[0, 1, 2, 3, 4], nrows=None)
PD.columns = ['pid', 'sex', 'birthdate', 'status', 'status_date']

# %%
PD.head()

# %% [markdown]
# **DNPR** 
# - only primary and secondary diagnoses
# - PID, OWNER, HOSPITAL_ID, WARD_ID, icd10, date, CODE_TYPE (dropped)

# %%
# loading data files
file = open(d_DNPR, 'r')
reg = pd.read_csv(file, sep='\t', usecols=[0, 4, 6, 7, 8, 9, 10], iterator=True, chunksize=250000)
DNPR = next(reg)

# %%
DNPR = adjust_data(DNPR, relevant_diagnosis)
n = 0
while True:
    try:
        reg_ = next(reg)
        reg_ = adjust_data(reg_, relevant_diagnosis)
        DNPR = DNPR.append(reg_)
    except:
        break

# %%
DNPR['count'] = 1
DNPR.columns = ['pid', 'date', 'public', 'hospital', 'ward', 'code', 'icd10', 'count']
DNPR.sort_values(['pid', 'date'], inplace=True)
DNPR = DNPR.groupby(['pid', 'icd10']).aggregate({'date': 'first', 
               'public': 'first', 
               'hospital': 'first', 
               'ward': 'first',
               'code': 'first',
               'count': 'sum'}).reset_index()
DNPR.loc[:, 'public'] = np.asarray(DNPR.loc[:, 'public'] == 'PUBLIC').astype(int)
DNPR.sort_values(['pid', 'date'], inplace=True)
DNPR = DNPR[['pid', 'date', 'icd10', 'public', 'hospital', 'ward', 'code', 'count']]

# %%
DNPR.head()

# %% [markdown]
# **CR**

# %%
# loading data
CR = pd.read_csv(open(d_CR, 'r'), sep='\t', usecols=[0, 2, 3, 4, 5], nrows=None)

# %%
# data adjustments
CR['C_ICD10'] = np.asarray( CR['C_ICD10']).astype(str)
CR['icd10'] = CR['C_ICD10'].apply(lambda x: x[0:-1])
CR = CR.loc[CR['icd10'].apply(lambda x: x in relevant_diagnosis)].reset_index(drop=True)


CR.columns = ['pid', 'date', 'tumour_count', 'code', 'morph', 'icd10']
CR.sort_values(['pid', 'date'], inplace=True) 
CR = CR[['pid', 'date', 'tumour_count', 'code', 'morph', 'icd10']]

# %%
np.min(CR['D_DIAGNOSEDATO'])

# %% [markdown]
# **DR**
# 
# 

# %%
# loading data
DR1 = pd.read_csv(open(d_DR1, 'r'), sep='\t', usecols=[0, 1, 2, 3, 4, 5], nrows=None)

# %%
# rename columns + mapping
DR1.columns = ['pid', 'date', 'code1', 'code2', 'code3', 'code4']
n = 1
for ii in ['code1', 'code2', 'code3', 'code4']:
    DR1[ii] = np.asarray(DR1[ii]).astype(str)
    DR1['icd10_' + str(n)] = DR1[ii].apply(lambda x: x[0:-1])
    DR1['icd10_' + str(n)].replace(mapping, inplace=True)
    DR1.loc[DR1['icd10_' + str(n)].apply(lambda x: x[:3] not in relevant_diagnosis), 'icd10_' + str(n)] = ''
    n += 1
DR1 = DR1.loc[np.logical_or.reduce((DR1['icd10_1'].apply(lambda x: x[:3] in relevant_diagnosis), DR1['icd10_2'].apply(lambda x: x[:3] in relevant_diagnosis), DR1['icd10_3'].apply(lambda x: x[:3] in relevant_diagnosis), DR1['icd10_4'].apply(lambda x: x[:3] in relevant_diagnosis)))].reset_index(drop=True)
DR1.replace({'0000': '', 
             'nan': ''}, inplace=True)

# %%
DR1.head()

# %%
DR2 = pd.read_csv(open(d_DR2, 'r'), sep='\t', usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], nrows=None)

# %%
DR2.columns = ['pid', 'date', 'code1', 'code2',
       'code3', 'code4', 'code5', 'code6', 'code7', 'code8',
       'code9', 'code10', 'code11', 'code12', 'code13']
n = 1
for ii in ['code1', 'code2',
       'code3', 'code4', 'code5', 'code6', 'code7', 'code8',
       'code9', 'code10', 'code11', 'code12', 'code13']:
    DR2.loc[:, ii] = np.asarray(DR2[ii]).astype(str)
    DR2['icd10_' + str(n)] = DR2[ii].apply(lambda x: x[0:-1])
    DR2.loc[DR2['icd10_' + str(n)].apply(lambda x: x not in relevant_diagnosis), 'icd10_' + str(n)] = ''
    n += 1
DR2 = DR2.loc[np.logical_or.reduce((DR2['icd10_1'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_2'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_3'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_4'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_5'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_6'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_7'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_8'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_9'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_10'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_11'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_12'].apply(lambda x: x in relevant_diagnosis), DR2['icd10_13'].apply(lambda x: x in relevant_diagnosis)))].reset_index(drop=True)
DR2.replace({'0000': '', 
             'nan': ''}, inplace=True)

# %%
DR2.head()

# %%
DR = DR2.append(DR1, sort=False)
DR = DR[['pid', 'date', 'icd10_1', 'icd10_2', 'icd10_3', 'icd10_4', 'icd10_5', 'icd10_6', 'icd10_7', 'icd10_8', 'icd10_9', 'icd10_10', 'icd10_11', 'icd10_12', 'icd10_13', 'code1', 'code2',
       'code3', 'code4', 'code5', 'code6', 'code7', 'code8',
       'code9', 'code10', 'code11', 'code12', 'code13']]

# %%
DR.head()

# %%
idx_list = np.unique(np.concatenate((np.asarray(DR['pid']), np.asarray(CR['pid']), np.asarray(DNPR['pid'])), axis=0))

# %%
print(DNPR.shape)
print(CR.shape)
print(DR.shape)
print(PD.shape)

# %% [markdown]
# #### Writing data:

# %%

nn = 0
step = int(np.ceil(idx_list.shape[0]/100))
with h5py.File(d_data + 'DB/cancer/_' + str(ind) + '.h5', 'w') as f:
    for idx in idx_list[ind * step: (ind + 1) * step]:  
        nn += 1 
        if nn % 500 == 0:
            file = open('/home/people/alexwolf/projects/DB/logs/progress.txt', 'w') 
            file.write(str(step) + ' : ' + str(nn)) 
            file.close()  
        f.create_group(idx)
        f[idx].create_dataset("DNPR", data=np.asarray(DNPR.loc[DNPR.iloc[:, 0] == idx])[:, 1:].astype('S10'), maxshape=(None, 7), compression="gzip", compression_opts=9)
        f[idx].create_dataset("CR", data=np.asarray(CR.loc[CR.iloc[:, 0] == idx])[:, 1:].astype('S10'), maxshape=(None, 5), compression="gzip", compression_opts=9)
        f[idx].create_dataset("DR", data=np.asarray(DR.loc[DR.iloc[:, 0] == idx])[:, 1:].astype('S10'), maxshape=(None, 27), compression="gzip", compression_opts=9)
        f[idx].create_dataset("PD", data=np.asarray(PD.loc[PD.iloc[:, 0] == idx])[:, 1:].astype('S10'), maxshape=(None, 4), compression="gzip", compression_opts=9)

# %%
print('finsihed')

# %% [markdown]
# ## Testing:

# %%
n1 = 0
n2 = 0
n3 = 0
for ind in np.arange(101):
    print(ind)
    with h5py.File(d_data + 'DB/cancer/_' + str(ind) + '.h5', 'r') as f:
        for ii in list(f.keys()):
            if f[ii]['DNPR'].shape[0] != 0:
                n1 += 1
            if f[ii]['CR'].shape[0] != 0:
                n2 += 1
            if f[ii]['DR'].shape[0] != 0:
                n3 += 1

        

# %% [markdown]
# ### Reference File:

# %%
key = []
value = []
for ind in np.arange(101):
    print(ind)
    with h5py.File(d_data + 'DB/cancer/_' + str(ind) + '.h5', 'r') as f:
        key.append(ind)
        value.append(list(f.keys()))


# %%
dicct = dict(zip(key, value))

# %%
pickle.dump(dicct, open(d_data + 'DB/cancer/_ref.pickle', "wb"))

# %%
# loading data
CR = pd.read_csv(open(d_CR, 'r'), sep='\t', usecols=[0, 2, 3, 4, 5], nrows=None)

# %%
CR = pd.read_csv('/home/projects/cancer_diabetes_lpr/new_preprocessing_jan2018/prepared_data/t_tumor.tsv', sep='\t')

# %%
np.min(CR['D_DIAGNOSEDATO'])

# %%
np.min(DR1['D_DODSDTO'])


