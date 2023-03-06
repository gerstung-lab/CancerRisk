# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import dill as pickle
import dill
import tqdm
import shutil

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
pyro.clear_param_store()

import warnings
warnings.filterwarnings("ignore")

# model - ProbCox
sys.path.append('/users/projects/cancer_risk/main/scripts/ProbCox')
import probcox as pcox 
from _custom_functions import CIF

# predictor
from m1 import predictor 

# dataloader
sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from dataloader import Data_Pipeline
from pipe_cancer import pipe_cancer

# help functions
from helper import custom_collate, IterativeSampler, RandomSampler

# seeding
np.random.seed(7)
torch.random.manual_seed(42)
pyro.set_rng_seed(9)

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'
root_dir = '/users/projects/cancer_risk/'

# train/valid/test
with open(dir_DB + 'trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)
    
# sample info
with open(dir_data + 'CancerRisk/samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)

# disease codes
disease_ref= pd.read_csv(dir_data + 'CancerRisk/disease_ref.csv', sep=';')
disease_ref = np.asarray(disease_ref.iloc[:, 2])
disease_cat = np.load(dir_data + 'CancerRisk/disease_codes_cat.npy')

dtype = torch.FloatTensor 

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']
run_id = int(sys.argv[1])
#run_id = 0

# %%
''' 
def makedir(x):
    try:
        os.mkdir(x)
    except:
        pass
    
for ii0 in range(10):
    makedir('/users/projects/cancer_risk/data/frequency/f_%i/' %(ii0))
    for ii1 in range(10):
        makedir('/users/projects/cancer_risk/data/frequency/f_%i/f_%i/' %(ii0, ii1))
'''

# %%
# Pipe 
# ==========================================================================================
# ==========================================================================================
PIPE = Data_Pipeline(event_idx=None, sex_specific=np.asarray([0, 1]), inference=False)
PIPE_cancer = pipe_cancer()

# %%
# Main 
# ==========================================================================================
# ==========================================================================================

for ii in [run_id]:
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    file = dir_DB + 'f_%i/f_%i/_%i' %(ii1, ii2, ii)
    file_out = dir_data + 'frequency/f_%i/f_%i/_%i' %(ii1, ii2, ii)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        with h5py.File(file_out, 'w') as fout:
            idx_list_out = list(fout.keys()) #r
            for jj in tqdm.tqdm(samples[ii]):
                if idx_list[jj] not in idx_list_out:
                    fidx = f[idx_list[jj]]
                    eid = idx_list[jj]
                    time, dnpr_dummies, genealogy, bth, meta = PIPE.__infer__(fidx=fidx)
                    birthdate, sex, status, EOO = [meta[0], meta[1], meta[2], meta[3]]
                    dates_cancer, decision, cancer_set = PIPE_cancer.__cancercall__(fidx=fidx)

                    if dates_cancer:
                        if dates_cancer <= np.datetime64('2015-01-01'):
                            continue

                    if EOO <= np.datetime64('2015-01-01'):
                        continue

                    if (np.datetime64('2015-01-01') - birthdate).astype(int) > 365*75:
                        continue

                    if (np.datetime64('2015-01-01') - birthdate).astype(float) < 16*365:
                        continue


                    fout.create_group(str(eid))
                    fout[str(eid)].create_dataset('x', data=dnpr_dummies[-1, :][None, :].astype(float), maxshape=(1, 1305), compression='lzf')



# %%
exit()

# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N freq
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=3gb
#PBS -l walltime=1:00:00

cd $PBS_O_WORDIR
module load anaconda3/2019.10
source conda activate

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/preprocessing/06_frequency.ipynb --output /users/projects/cancer_risk/main/scripts/preprocessing/06_frequency

/services/tools/anaconda3/2019.10/bin/python3.7 /users/projects/cancer_risk/main/scripts/preprocessing/06_frequency.py $VAR1
' >> run.sh

for ii in 948; do qsub -v VAR1=$ii run.sh; done


# %%
# Combine 
# ==========================================================================================
# ==========================================================================================
nn = 0 
file_out = dir_data + '/frequency/master.h5'
with h5py.File(file_out, 'w') as fout:
    fout.create_dataset("data", (4248491, 1305), dtype='float')
    n = 0
    for ii in tqdm.tqdm(range(1000)):
        ii1 = ii//100
        ii2 = (ii - ii1*100)//10
        file = dir_data + 'frequency/f_%i/f_%i/_%i' %(ii1, ii2, ii)
        with h5py.File(file, 'r') as f:
            idx_list = list(f.keys())
            nn += len(idx_list)
            for eid in idx_list: 
                fout['data'][n, :] = np.minimum(f[str(eid)]['x'][:, :].astype(float), 1.0)
                n += 1
print(nn)

# %%
print(nn)

# %%

ll =[]
n = 0 
for ii in tqdm.tqdm([33, 570]):
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    file1 = dir_data + 'predictions_ukb/f_%i/f_%i/s_%i' %(ii1, ii2, ii)
    file2 = dir_data + 'frequency/f_%i/f_%i/_%i' %(ii1, ii2, ii)
    try:
        with h5py.File(file1, 'r') as f1:
            with h5py.File(file2, 'r') as f2:
                if len(list(f1.keys())) != len(list(f2.keys())):
                    ll.extend([ii])
    except:
        ll.extend([ii])


# %%

n = 0 
for ii in tqdm.tqdm(range(500, 1000)):
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    file_out = dir_data + 'frequency/f_%i/f_%i/_%i' %(ii1, ii2, ii)
    with h5py.File(file_out, 'r') as fout:
        n += len(list(fout.keys())) #r
print(n)


# %%
'''
ll =[]
n = 0 
for ii in tqdm.tqdm(range(1000)):
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    #file1 = dir_data + 'predictions_ukb/f_%i/f_%i/s_%i' %(ii1, ii2, ii)
    file2 = dir_data + 'frequency/f_%i/f_%i/_%i' %(ii1, ii2, ii)
    try:
        with h5py.File(file2, 'r') as f2:
            pass
    except:
        ll.extend([ii])
ss=''
for ii in ll:
    ss += str(ii) + ' '
ss
'''


