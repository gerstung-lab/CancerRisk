# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import pickle
import dill
import tqdm

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

import warnings
warnings.filterwarnings("ignore")

# model - ProbCox
sys.path.append('/home/people/alexwolf/projects/CancerRisk/scripts/ProbCox')
import probcox as pcox 

# predictor
#from m1 import predictor 

# dataloader
sys.path.append('/home/people/alexwolf/projects/CancerRisk/scripts/dataloader')
from dataloader import Data_Pipeline

# help functions
from helper import custom_collate, IterativeSampler, RandomSampler, concordance_plot

# seeding
np.random.seed(11)
torch.random.manual_seed(12)
pyro.set_rng_seed(13)


# directories 
dir_data = '/home/people/alexwolf/data/'
dir_DB = '/home/people/alexwolf/data/DB/DB/raw/'
dir_out = '/home/people/alexwolf/projects/CancerRisk/output/'

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver',
                'Pancreas', 'Lung', 'Melanoma', 'Breast', 'Cervix uteri',
                'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney',
                'Bladder', 'Brain', 'Thyroid', 'Non Hodgkin Lymphoma', 'Multiple Myeloma', 'AML', 'Other', 'Death']

# train/valid/test
with open(dir_DB + 'trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)
    
    
# cluster
#event_idx = int(sys.argv[1])
#event_idx=21

# %%
# Custom funtions
# =======================================================================================================================
class A0_fun():
    def __init__(self, tt, basehaz):
        self.tt = tt
        self.basehaz = basehaz
        
    def __call__(self, ii):
        
        if np.sum(ii > self.tt) == 0:
            return(0)
        elif np.sum(ii > self.tt) <= len(self.tt):
            return(self.basehaz[np.sum(ii > self.tt)-1][0])
        else:
            return(self.basehaz[-1][0])

# %%
# Data
# =======================================================================================================================
time = []
pred = []
for run_id in tqdm.tqdm(tvt_split['train']):#
    ii1 = run_id//100
    ii2 = (run_id - ii1*100)//10
    file = dir_out + 'predictions/f_%i/f_%i/_%i.h5' %(ii1, ii2, run_id)
    with h5py.File(file, 'r') as f:
        time.extend(f['main'][:, (2, 3, 6)].tolist())
        pred.extend(f['pred'][:, (0, 2), event_idx].tolist())

time = np.asarray(time)
pred = np.asarray(pred)
sex = time[:, -1].astype(bool)
time = time[:, :-1]

# %%
# Breslow - CIF
# =======================================================================================================================

tt, basehaz = pcox.Breslow(times=np.concatenate((time[~sex, :], pred[~sex, 0, None]), axis=1), pred=pred[~sex, 1, None])    
delta_time =[]
for jj in np.arange(0, tt.shape[0]-1):
    delta_time.append(tt[jj+1] - tt[jj])
delta_time.append(0.1)
delta_time = np.asarray(delta_time)[:, None]
delta_time = np.asarray([np.sum(delta_time[jj==tt], axis=0) for jj in np.unique(tt)])
basehaz = np.asarray([np.sum(basehaz[jj==tt], axis=0) for jj in np.unique(tt)])[:, None]
tt = np.unique(tt)
basehaz = basehaz/delta_time
A0_f = A0_fun(tt=tt[:-2], basehaz=basehaz[:-2])

tt, basehaz = pcox.Breslow(times=np.concatenate((time[sex, :]-100000, pred[sex, 0, None]), axis=1), pred=pred[sex, 1, None])
delta_time =[]
for jj in np.arange(0, tt.shape[0]-1):
    delta_time.append(tt[jj+1] - tt[jj])
delta_time.append(0.1)
delta_time = np.asarray(delta_time)[:, None]
delta_time = np.asarray([np.sum(delta_time[jj==tt], axis=0) for jj in np.unique(tt)])
basehaz = np.asarray([np.sum(basehaz[jj==tt], axis=0) for jj in np.unique(tt)])[:, None]
tt = np.unique(tt)
basehaz = basehaz/delta_time
A0_m = A0_fun(tt=tt[:-2], basehaz=basehaz[:-2])

dill.dump({'female':A0_f, 'male':A0_m}, open(dir_out + 'model/' + events[event_idx] + '/breslow.pkl', 'wb'))              

# %%
exit()

# %%
%%bash

rm /home/people/alexwolf/run.sh

ssh precision05
echo '
#!/bin/sh
#PBS -N breslow
#PBS -o /home/people/alexwolf/_/
#PBS -e /home/people/alexwolf/_/
#PBS -l nodes=1:ppn=4
#PBS -l mem=16gb
#PBS -l walltime=300:00:00

cd $PBS_O_WORDIR
module load tools
module load anaconda3/5.3.0
source conda activate

jupyter nbconvert --to script /home/people/alexwolf/projects/CancerRisk/scripts/m1/breslow.ipynb --output /home/people/alexwolf/projects/CancerRisk/scripts/m1/breslow

/services/tools/anaconda3/5.3.0/bin/python3.6 /home/people/alexwolf/projects/CancerRisk/scripts/m1/breslow.py $VAR1
' >> run.sh


for ii in {0..22}; do sleep 1; qsub -v VAR1=$ii run.sh; done



# %%
# Transform pickled object to matrix

# %%
A0 = [dill.load(open(dir_out + 'model/' + events[event_idx] + '/breslow.pkl', 'rb')) for event_idx in range(22)]


# %%
AA = np.stack([[[A0[cc]['female'](ii) for ii in range(31400)], [A0[cc]['male'](ii) for ii in range(31400)]] for cc in tqdm.tqdm(range(22))])
AA[7, 1, :] = 0  # male breast cancer to 0

# %%
np.save(dir_out + 'model/' + 'all' + '/breslow.npy', AA)

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



