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

with open(dir_data + 'CancerRisk/samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)
    
disease_codes = np.load(dir_data + 'CancerRisk/disease_codes.npy', allow_pickle=True)

# train/valid/test
with open(dir_data + 'DB/DB/raw/trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)

# %%
# Pipe 
# ==========================================================================================
# ==========================================================================================

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_cancer import pipe_cancer

PIPE = pipe_cancer()


# %%
locations = [[] for _ in range(22)]
cancer_f = []
cancer_m = []
n_m = 0
n_f = 0
for hh in tqdm.tqdm(tvt_split['train']): # tvt_split['train']
    ii1 = hh//100
    ii2 = (hh - ii1*100)//10
    file = root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, hh)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for ii in samples[hh]:
            fidx = f[idx_list[ii]]
            dates_cancer, decision, _ = PIPE.__cancercall__(fidx=fidx)
            X = PIPE.__maincall__(fidx=fidx)
            if X[1] == 0:
                n_f += 1
            elif X[1] == 1:
                n_m += 1
            
            if dates_cancer:
                if dates_cancer > np.datetime64('2014-12-31'):
                    continue
                if (dates_cancer - X[0]).astype(int) > 365*86:
                    continue
                    
            if np.any(decision != None): 
                decision = np.concatenate((decision, np.zeros((1,)).astype(float)))
            elif np.logical_and(X[3] <= np.datetime64('2014-12-31'), (X[3] - X[0]).astype(int) <= 365*86):
                decision = np.zeros((21, )).astype(float)
                decision = np.concatenate((decision, (X[2]==90).astype(float)))
            else:
                decision = np.zeros((22, )).astype(float)
                
            decision[7] = decision[7] * (1-X[1]) # only female breast cancer
            for ll in np.where(decision)[0]:
                locations[ll].append([hh, ii])
            # sex split
            if X[1] == 0:
                if np.any(decision != None):
                    cancer_f.extend([decision.tolist()])
            elif X[1] == 1:
                if np.any(decision != None):
                    cancer_m.extend([decision.tolist()])
    cancer_f = np.sum(cancer_f, axis=0)[None, :].tolist()
    cancer_m = np.sum(cancer_m, axis=0)[None, :].tolist()
print(n_f, n_m)

# %%
with open(dir_out + 'locations2.pickle', 'wb') as handle:
    pickle.dump(locations, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
with open(dir_out + 'proportions.pickle', 'wb') as handle:
    pickle.dump([n_f, [cancer_f[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21]], n_m, [cancer_m[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]], handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
ll = {}
for cc in range(22):
    ll[cc] = {}
    for ii in tvt_split['train']:
        ll[cc][ii] = []
    for ii in locations[cc]:
        ll[cc][ii[0]].extend([ii[1]])
    
with open(dir_out + 'locations.pickle', 'wb') as handle:
    pickle.dump(ll, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
locations = [[] for _ in range(22)]
cancer_f = []
cancer_m = []
n_m = 0
n_f = 0
for hh in tqdm.tqdm(tvt_split['valid']): # tvt_split['train']
    ii1 = hh//100
    ii2 = (hh - ii1*100)//10
    file = root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, hh)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for ii in samples[hh]:
            fidx = f[idx_list[ii]]
            dates_cancer, decision, _ = PIPE.__cancercall__(fidx=fidx)
            X = PIPE.__maincall__(fidx=fidx)
            if X[1] == 0:
                n_f += 1
            elif X[1] == 1:
                n_m += 1
            
            if dates_cancer:
                if dates_cancer > np.datetime64('2014-12-31'):
                    continue
                if (dates_cancer - X[0]).astype(int) > 365*86:
                    continue
            
            if np.any(decision != None): 
                decision = np.concatenate((decision, np.zeros((1,)).astype(float)))
            elif np.logical_and(X[3] <= np.datetime64('2014-12-31'), (X[3] - X[0]).astype(int) < 365*86):
                decision = np.zeros((21, )).astype(float)
                decision = np.concatenate((decision, (X[2]==90).astype(float)))
            else:
                decision = np.zeros((22, )).astype(float)
                
            decision[7] = decision[7] * (1-X[1]) # only female breast cancer
            for ll in np.where(decision)[0]:
                locations[ll].append([hh, ii])
            # sex split
            if X[1] == 0:
                if np.any(decision != None):
                    cancer_f.extend([decision.tolist()])
            elif X[1] == 1:
                if np.any(decision != None):
                    cancer_m.extend([decision.tolist()])
    cancer_f = np.sum(cancer_f, axis=0)[None, :].tolist()
    cancer_m = np.sum(cancer_m, axis=0)[None, :].tolist()
print(n_f, n_m)

with open(dir_out + 'proportions_valid.pickle', 'wb') as handle:
    pickle.dump([n_f, [cancer_f[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21]], n_m, [cancer_m[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]], handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
locations = [[] for _ in range(22)]
cancer_f = []
cancer_m = []
n_m = 0
n_f = 0
for hh in tqdm.tqdm(tvt_split['test']): # tvt_split['train']
    ii1 = hh//100
    ii2 = (hh - ii1*100)//10
    file = root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, hh)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for ii in samples[hh]:
            fidx = f[idx_list[ii]]
            dates_cancer, decision, _ = PIPE.__cancercall__(fidx=fidx)
            X = PIPE.__maincall__(fidx=fidx)
            if X[1] == 0:
                n_f += 1
            elif X[1] == 1:
                n_m += 1
            
            if dates_cancer:
                if dates_cancer > np.datetime64('2014-12-31'):
                    continue
                if (dates_cancer - X[0]).astype(int) > 365*86:
                    continue
                
            if np.any(decision != None): 
                decision = np.concatenate((decision, np.zeros((1,)).astype(float)))
            elif np.logical_and(X[3] <= np.datetime64('2014-12-31'), (X[3] - X[0]).astype(int) < 365*86):
                decision = np.zeros((21, )).astype(float)
                decision = np.concatenate((decision, (X[2]==90).astype(float)))
            else:
                decision = np.zeros((22, )).astype(float)
                
            decision[7] = decision[7] * (1-X[1]) # only female breast cancer
            for ll in np.where(decision)[0]:
                locations[ll].append([hh, ii])
            # sex split
            if X[1] == 0:
                if np.any(decision != None):
                    cancer_f.extend([decision.tolist()])
            elif X[1] == 1:
                if np.any(decision != None):
                    cancer_m.extend([decision.tolist()])
    cancer_f = np.sum(cancer_f, axis=0)[None, :].tolist()
    cancer_m = np.sum(cancer_m, axis=0)[None, :].tolist()
print(n_f, n_m)

with open(dir_out + 'proportions_test.pickle', 'wb') as handle:
    pickle.dump([n_f, [cancer_f[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21]], n_m, [cancer_m[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]], handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
locations = [[] for _ in range(22)]
cancer_f = []
cancer_m = []
n_m = 0
n_f = 0
for hh in tqdm.tqdm(range(1000)): 
    ii1 = hh//100
    ii2 = (hh - ii1*100)//10
    file = dir_DB + 'f_%i/f_%i/_%i' %(ii1, ii2, hh)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for ii in samples[hh]:
            fidx = f[idx_list[ii]]
            dates_cancer, decision, _ = PIPE.__cancercall__(fidx=fidx)
            X = PIPE.__maincall__(fidx=fidx) # birthdate, sex, status, EOO, SOO, idxcancer
            
            if dates_cancer:
                if dates_cancer <= np.datetime64('2015-01-01'):
                    continue
                    
            if X[3] <= np.datetime64('2015-01-01'):
                continue
            
            if (np.datetime64('2015-01-01') - X[0]).astype(int) > 365*75:
                continue
                
            if (np.datetime64('2015-01-01') - X[0]).astype(float) < 16*365:
                continue
                
            if X[1] == 0:
                n_f += 1
            elif X[1] == 1:
                n_m += 1
                
            if np.any(decision != None): 
                decision = np.concatenate((decision, np.zeros((1,)).astype(float)))
            else:
                decision = np.zeros((21, )).astype(float)
                decision = np.concatenate((decision, (X[2]==90).astype(float)))

            decision[7] = decision[7] * (1-X[1]) # only female breast cancer
            for ll in np.where(decision)[0]:
                locations[ll].append([hh, ii])
                
            # sex split
            if X[1] == 0:
                if np.any(decision != None):
                    cancer_f.extend([decision.tolist()])
            elif X[1] == 1:
                if np.any(decision != None):
                    cancer_m.extend([decision.tolist()])
    cancer_f = np.sum(cancer_f, axis=0)[None, :].tolist()
    cancer_m = np.sum(cancer_m, axis=0)[None, :].tolist()
print(n_f, n_m)

with open(dir_data + '/CancerRisk/proportions_test2.pickle', 'wb') as handle:
    pickle.dump([n_f, [cancer_f[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21]], n_m, [cancer_m[0][ii] for ii in [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]]], handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
exit()

# %%
%%bash

rm /home/people/alexwolf/run.sh

ssh precision05
echo '
#!/bin/sh
#PBS -N cancer
#PBS -o /home/people/alexwolf/_/
#PBS -e /home/people/alexwolf/_/
#PBS -l nodes=1:ppn=4
#PBS -l mem=18gb
#PBS -l walltime=300:00:00

cd $PBS_O_WORDIR
module load tools
module load anaconda3/5.3.0
source conda activate

jupyter nbconvert --to script /home/people/alexwolf/projects/CancerRisk/scripts/preprocessing/03_cancer_cases.ipynb --output /home/people/alexwolf/projects/CancerRisk/scripts/preprocessing/03_cancer_cases

/services/tools/anaconda3/5.3.0/bin/python3.6 /home/people/alexwolf/projects/CancerRisk/scripts/preprocessing/03_cancer_cases.py 
' >> run.sh

qsub run.sh





