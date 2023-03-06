# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import pickle
import torch 

import tqdm

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from torch.utils.data import DataLoader
from multiprocessing import Pool

# seeding
np.random.seed(7)
torch.random.manual_seed(8)

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'
dir_out = '/users/projects/cancer_risk/data/CancerRisk/'

# relevant cancer codes - malignant/uncertain/treatment
malignant_codes = np.concatenate([['C0' + str(_) for _ in range(10)],
        ['C' + str(_) for _ in range(10,100)], 
        ['Z08', 'Z85']])

prior_cancer = np.concatenate([
        ['C' + str(_) for _ in range(77,80)], 
        ['Z08', 'Z85']])

uncertain_cancer = np.asarray(['D' + str(_) for _ in range(37,49)])

# %%
# Custom Functions
# ==========================================================================================
# ==========================================================================================

def __in__(x, y, reduce_x=False):
        """
        Doc: 
            Compares whether the entries from on list (main) appear in the other list (secondary)

        Args:
            x (list):                   Entires in main list
            y (list):                   Entires in secondary list
            reduce_x(bool):             Whether to cut of part of x ( -> 3rd level icd10 code)

        Return: 
            helpvar (np.bool)(1,):      Array with logical indicators  
        """
        x = np.asarray(x)
        y = np.asarray(y)
        helpvar = []
        if reduce_x:
            for ii in x:
                if len(ii) == 4:
                    helpvar.append(ii[:-1] in y)
                else:
                    helpvar.append(ii in y)   
        else:
            for ii in x: 
                helpvar.append(ii in y)
        return(np.asarray(helpvar))

# %%
# Pipe 
# ==========================================================================================
# ==========================================================================================

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')

from pipe_cancer import pipe_cancer
PIPE = pipe_cancer()

# %%
# Main
# ==========================================================================================
# ==========================================================================================
dicct = {}

n_1 = 0
n_2 = 0
n_3 = 0
n_4 = 0
n_5 = 0
n = 0
for ii1 in (range(10)):
    for ii2 in range(10):
        for ii3 in range(10):
            ii4=100*ii1 + 10*ii2 + ii3
            print(ii4)
            dicct[ii4] = []

            file = dir_DB + 'f_%i/f_%i/_%i' %(ii1, ii2, ii4)
            with h5py.File(file, 'r') as f:
                idx_list = list(f.keys())
                n += len(list(f.keys()))
                for ii in range(0,len(idx_list)):
                    fidx = f[idx_list[ii]]
                    birthdate, sex, status, EOO, SOO, idxcancer = PIPE.__maincall__(fidx=fidx)
                    
                    # death before 1995
                    if EOO <= np.datetime64('1995-01-01'):
                        n_1 += 1
                        continue
                        
                    # 85+ before 1994 - 1995 since we remove the first year prior to a diagnosis
                    if birthdate + 365*86 <= np.datetime64('1995-01-01'):
                        n_1 += 1
                        continue
                        
                    # min 16yrs old
                    if (EOO - birthdate).astype(int) < 365*16:
                        n_1 += 1
                        continue

                    cancer_set = PIPE.__cancer__(fidx=fidx)
                    cancer_set[:, 1] = np.asarray([kk[:4] for kk in cancer_set[:, 1]])
                    idx_cancer = __in__(cancer_set[:, 1], malignant_codes, reduce_x=True)
                    if np.sum(idx_cancer) > 0:
                        # subset to relevant codes 
                        primary = ((cancer_set[idx_cancer, 0].astype('datetime64[D]') - np.min(cancer_set[idx_cancer, 0].astype('datetime64[D]'))).astype(int) < 31*3)
                        date = np.min(cancer_set[idx_cancer, 0].astype('datetime64[D]'))
                        
                        # possibly adjustby uncertain cancer diagnosis date (if less then 1 year apart)
                        idx_uncertain = __in__(cancer_set[:, 1], uncertain_cancer, reduce_x=True)
                        if np.sum(idx_uncertain) > 0:
                            date_uncertain = np.min((cancer_set[idx_uncertain, 0]).astype('datetime64[D]'))
                            if (date - date_uncertain).astype(int) <= 365:  
                                date = np.minimum(date, date_uncertain)

                        # remove cancers before 1995 since we remove the first year prior to a diagnosis
                        if date <= np.datetime64('1995-01-01'):
                            n_2 += 1
                            continue

                        if (date - birthdate).astype(int) <= 365*16:
                            n_2 += 1
                            continue
                            
                        # remove cancers that have entries after EOO+1 month (migration etc.)
                        if date > EOO+31:
                            n_3 += 1
                            continue

                        cancer_set = cancer_set[idx_cancer, :]
                        cancer_set = cancer_set[primary, :]
                        idx_prior_cancer = __in__(cancer_set[:, 1], prior_cancer, reduce_x=True)
                        cancer_set = cancer_set[~idx_prior_cancer, :]

                        # remove if only secondary or treatment avail.
                        if cancer_set.shape[0] == 0:
                            n_4 += 1
                            continue

                    dicct[ii4].extend([ii])
                    n_5 += 1
        
      
#with open(dir_out + 'samples.pickle', 'wb') as handle:
#    pickle.dump(dicct, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
print(n_1)
print(n_2)
print(n_3)
print(n_4)
print(n_5)

# %%
exit()

# %%


# %%



