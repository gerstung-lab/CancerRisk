# %%
# Modules
#=========================================================================
#=========================================================================

import sys
import os 
import h5py
import pickle
import tqdm

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

dir_data = '/home/people/alexwolf/data/'

np.random.seed(7)

# icd10 ref
file = open('/home/classification/complete/icd10_eng_diag_chapters_all_update2016.csv', 'r')
disease_mapping_ref = pd.read_csv(file, sep='\t', header=None)
file.close()

# %%
disease_mapping_ref = disease_mapping_ref.iloc[:, [0, 1, 2]]

# %%
disease_mapping_ref = disease_mapping_ref.loc[np.asarray(disease_mapping_ref.iloc[:, 0].apply(lambda x: len(x))) == 3]

# %%
remove = [ 'S', 'T', 'V', 'X', 'Y', 'Z', 'U', 'C', 'W']
disease_mapping_ref = disease_mapping_ref.loc[np.asarray(disease_mapping_ref.iloc[:, 0].apply(lambda x: x[0] not in remove))]
disease_mapping_ref = disease_mapping_ref.reset_index(drop=True)
#remove = ['D' + str(ii) for ii in range(10, 37)]
#remove.extend(['D0' + str(ii) for ii in range(10)])
#disease_mapping_ref = disease_mapping_ref.loc[np.asarray(disease_mapping_ref.iloc[:, 0].apply(lambda x: x[:2] not in remove))]


# %%
disease_mapping_ref.to_csv(dir_data + 'CancerRisk/disease_ref.csv', sep=';')


# %%
disease_codes = np.asarray(disease_mapping_ref.iloc[:, 0])

# %%
np.save(dir_data + 'CancerRisk/disease_codes.npy', disease_codes)

# %%
d_start = [0]
d_end = []
d_cat = ['Certain infectious and parasitic diseases']

helpvar = np.asarray(disease_mapping_ref.iloc[:, -1])
for ii in range(1, disease_mapping_ref.shape[0]):
    if helpvar[ii] in d_cat:
        pass
    else:
        d_cat.append(helpvar[ii])
        d_start.append(ii)
        d_end.append(ii)
d_end.append(disease_mapping_ref.shape[0])

# %%
np.save(dir_data + 'CancerRisk/disease_codes_cat.npy', np.concatenate((np.asarray(d_start)[:, None], np.asarray(d_end)[:, None], np.asarray(d_cat)[:, None]), axis=1))

# %%
exit()

# %%
disease_mapping_ref.loc[:, [2, 3]].groupby(2).sum()

# %%



