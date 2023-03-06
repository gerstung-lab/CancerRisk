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
dir_data = '/home/people/alexwolf/data/'
dir_DB = '/home/people/alexwolf/data/DB/DB/raw/'
dir_out = '/home/people/alexwolf/data/CancerRisk/'
root_dir = '/home/people/alexwolf/data/DB/DB/raw/'

with open(dir_out + 'samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)

disease_codes = np.load(dir_data + 'CancerRisk/disease_codes.npy', allow_pickle=True)

# disease codes
disease_ref = pd.read_csv(dir_data + 'CancerRisk/disease_ref.csv', sep=';')
disease_ref = np.asarray(disease_ref.iloc[:, 2])

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver',
                'Pancreas', 'Lung', 'Melanoma', 'Breast', 'Cervix uteri',
                'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney',
                'Bladder', 'Brain', 'Thyroid', 'Non Hodgkin Lymphoma', 'Multiple Myeloma', 'AML', 'Other', 'Death']


# %%
# Pipe 
# ==========================================================================================
# ==========================================================================================

sys.path.append('/home/people/alexwolf/projects/CancerRisk/scripts/dataloader')

from dataloader import Data_Pipeline

PIPE = Data_Pipeline(event_idx=None, sex_specific=np.asarray([0, 1]))


# %%
# generate empty dicct
# hierarchy- 0, 1 (gender) - 1800, 2015 (year) - 0, 150 (age)
spop = {i : {j : {k: 0 for k in np.arange(0, 150)} for j in np.arange(1800, 2016)} for i in [0, 1]}
    
res_cancer = {}
for cc in Events:
    res_cancer[cc] = []


# %%
for hh in tqdm.tqdm(range(1000)): # tvt_split['train']
    ii1 = hh//100
    ii2 = (hh - ii1*100)//10
    file = root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, hh)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for ii in (samples[hh]):
            fidx = f[idx_list[ii]]
            
            dates_cancer, decision, _ = PIPE.__cancercall__(fidx=fidx)
            birthdate, sex, status, EOO, SOO, idxcancer = PIPE.__maincall__(fidx=fidx)
            EOO = np.minimum(EOO, np.datetime64('2015-01-01'))
            
            years = np.arange(int(np.maximum(SOO, birthdate).astype(str)[0][:4]), int(EOO.astype(str)[0][:4]))
            age = np.maximum(0, np.floor((SOO-birthdate).astype(float)/365)) + np.arange(years.shape[0])
            for kk in range(years.shape[0]):
                spop[sex[0]][years[kk]][age[kk]] += 1
            
            if dates_cancer:
                if dates_cancer > np.datetime64('2014-12-31'):
                    continue
                if (dates_cancer - birthdate).astype(int) > 365*86:
                    continue
                    
            if np.any(decision != None): 
                decision = np.concatenate((decision, np.zeros((1,)).astype(float)))
            elif np.logical_and(EOO <= np.datetime64('2014-12-31'), (EOO - birthdate).astype(int) <= 365*86):
                decision = np.zeros((21, )).astype(float)
                decision = np.concatenate((decision, (status==90).astype(float)))
            else:
                decision = np.zeros((22, )).astype(float)
                
            decision[7] = decision[7] * (1-sex) # only female breast cancer
            
            if decision.sum()>0:
                if dates_cancer:
                    dates_cancer = np.minimum(dates_cancer, EOO[0])
                else:
                    dates_cancer = EOO[0]
                for jj in np.where(decision)[0]:
                    res_cancer[Events[jj]].extend([np.concatenate((sex, np.floor((dates_cancer - birthdate).astype(float)/365), np.asarray([int(dates_cancer.astype(str)[:4])]))).tolist()])

            

# %%
# save file 
with open(dir_data + 'CancerRisk/standardized_population.pkl', 'wb') as f:
    pickle.dump(spop, f, pickle.HIGHEST_PROTOCOL)
    
# save file 
with open(dir_data + 'CancerRisk/cancer_incidence.pkl', 'wb') as f:
    pickle.dump(res_cancer, f, pickle.HIGHEST_PROTOCOL)

# %%
# save subset to Dataframe
year = []
age = []
sex = []
total = []
for i in [0, 1]:
    for j in  np.arange(1994, 2016):
         for k in np.arange(0, 150):
                sex.append(i)
                year.append(j)
                age.append(k)
                total.append(spop[i][j][k])
df = pd.DataFrame(list(zip(sex, year, age, total)), 
               columns =['sex', 'year', 'age', 'total']) 
df.to_csv(dir_data + 'CancerRisk/standardized_population.csv', sep=';')

# %%
exit()


