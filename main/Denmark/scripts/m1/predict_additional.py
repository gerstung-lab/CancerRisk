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
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
pyro.clear_param_store()

import warnings
warnings.filterwarnings("ignore")

# model - ProbCox
sys.path.append('/home/people/alexwolf/projects/CancerRisk/scripts/ProbCox')
import probcox as pcox 

# predictor
from m1 import predictor 

# dataloader
sys.path.append('/home/people/alexwolf/projects/CancerRisk/scripts/dataloader')
from dataloader import Data_Pipeline
from pipe_cancer import pipe_cancer

# help functions
from helper import custom_collate, IterativeSampler, RandomSampler

# seeding
np.random.seed(7)
torch.random.manual_seed(42)
pyro.set_rng_seed(9)

# directories 
dir_data = '/home/people/alexwolf/data/'
dir_DB = '/home/people/alexwolf/data/DB/DB/raw/'
dir_out = '/home/people/alexwolf/projects/CancerRisk/output/'
dir_logs = '/home/people/alexwolf/projects/CancerRisk/logs/'
root_dir = '/home/people/alexwolf/data/DB/DB/raw/'

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
# Pipe 
# ==========================================================================================
# ==========================================================================================
PIPE = Data_Pipeline(event_idx=None, sex_specific=np.asarray([0, 1]), inference=False)
PIPE_cancer = pipe_cancer()

# %%
# Models 
# ==========================================================================================
# ==========================================================================================
tt = [pickle.load(open(dir_out + 'model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]

# %%
# Main 
# ==========================================================================================
# ==========================================================================================
ll_main = []
ll_pred = []
ll_post = []
ll_post2 = []
ll_post3 = []

for ii in [run_id]:
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    file = root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, ii)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for jj in tqdm.tqdm(samples[ii]):
            fidx = f[idx_list[jj]]
            time, dnpr_dummies, genealogy, bth, meta = PIPE.__infer__(fidx=fidx)
            dates_cancer, decision, cancer_set = PIPE_cancer.__cancercall__(fidx=fidx)
            pp =[]
            for cc in range(len(tt)):
                pyro.clear_param_store()
                with torch.no_grad(): 
                    mm = tt[cc]['model']
                    gg = tt[cc]['guide']
                    theta_dnpr = tt[cc]['guide'].quantiles([0.5])['theta_dnpr'][0].detach().numpy()
                    theta_gene = tt[cc]['guide'].quantiles([0.5])['theta_gene'][0].detach().numpy()
                    theta_bth = tt[cc]['guide'].quantiles([0.5])['theta_bth'][0].detach().numpy()

                    p_dnpr = np.matmul(dnpr_dummies, theta_dnpr.T)
                    p_dnpr_sub = np.concatenate([np.matmul(dnpr_dummies[:, disease_cat[cat, 0].astype(int):disease_cat[cat, 1].astype(int)], theta_dnpr.T[disease_cat[cat, 0].astype(int):disease_cat[cat, 1].astype(int)])
                 for cat in range(18)], axis=1)
                    p_gene = np.matmul(genealogy, theta_gene.T)
                    p_bth = np.matmul(bth, theta_bth.T)

                    pred = p_dnpr + p_gene + p_bth 
                    
                    if pred.shape[0]==0:
                        pred = np.ones((time.shape[0], 1)) * -9999 # adding non sex prediction
                    pred_helpvar = np.concatenate((pred, p_dnpr, p_gene, p_bth, p_dnpr_sub), axis=1)                
                    pp.append(pred_helpvar)   
  
            ll_pred.extend(np.concatenate((time[:, None, 2:], np.concatenate([np.max(time[:, 2:], axis=0)[None, :] for _ in range(time.shape[0])], axis=0)[:, None, :], np.stack(pp, axis=-1)), axis=1).tolist()) 
            rep = time.shape[0]
            a = np.concatenate((np.repeat(ii, rep)[:, None].astype(float), #file
            np.repeat(jj, rep)[:, None].astype(float), #location
            time[:, :2].astype(float), 
            (np.max(time[:, 1]) - time[:, 0])[:, None], #time to last
            np.repeat(meta[0], rep)[:, None].astype(float),  #birthdate
            np.repeat(meta[1], rep)[:, None].astype(float),  #sex
            np.repeat(meta[2], rep)[:, None].astype(float),  #status
            np.repeat(meta[3], rep)[:, None].astype(float),  #EOO
            ), axis=1)                   
            ll_main.extend(a.tolist())
            
            if np.logical_and(np.max(time[:, 2:]) == 0, meta[3] >= np.asarray(['2015-01-01']).astype('datetime64[D]')):
            
                if np.any(decision==None):
                    dates_cancer = np.asarray(-9999.)
                    decision = np.repeat(0., 21)
                decision = np.concatenate((decision[None, :], (meta[3] < np.asarray(['2018-04-10']).astype('datetime64[D]')).astype(float)[None, :]), axis=1)
                b = np.concatenate((np.asarray([ii])[:, None].astype(float), #file
                np.asarray([jj])[:, None].astype(float), #location
                dates_cancer[None, None].astype(float),    
                (np.asarray(['2014-01-01']).astype('datetime64[D]') - meta[0]).astype(float)[:, None], # age 2014
                np.asarray([meta[1]]).astype(float), #sex 
                np.asarray([meta[2]]).astype(float), #status
                np.asarray([meta[3]]).astype(float)  #EOO
                ), axis=1)                   
                ll_post.extend(b.tolist())
                ll_post2.extend(np.concatenate((decision[:, None, :], np.stack(pp, axis=-1)[-1:, :1, :]), axis=1).tolist()) 
                ll_post3.extend(np.stack(pp, axis=-1)[-1:, 1:4, :].tolist()) 


# %%
ll_main = np.asarray(ll_main)
ll_pred = np.asarray(ll_pred)
ll_post = np.asarray(ll_post)
ll_post2 = np.asarray(ll_post2)
ll_post3 = np.asarray(ll_post3)

# %%
ii1 = run_id//100
ii2 = (run_id - ii1*100)//10
file = dir_out + 'predictions/f_%i/f_%i/_%i.h5' %(ii1, ii2, run_id)

with h5py.File(file, 'a') as f:
        f.create_dataset('postpred_additional', data=ll_post3, maxshape=(None, 3, 22), compression='lzf')

# %%
exit()

# %%
%%bash

rm /home/people/alexwolf/run.sh

ssh precision05
echo '
#!/bin/sh
#PBS -N predict
#PBS -o /home/people/alexwolf/_/
#PBS -e /home/people/alexwolf/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=4gb
#PBS -l walltime=1:00:00

cd $PBS_O_WORDIR
module load tools
module load anaconda3/5.3.0
source conda activate

jupyter nbconvert --to script /home/people/alexwolf/projects/CancerRisk/scripts/m1/predict_additional.ipynb --output /home/people/alexwolf/projects/CancerRisk/scripts/m1/predict_additional

/services/tools/anaconda3/5.3.0/bin/python3.6 /home/people/alexwolf/projects/CancerRisk/scripts/m1/predict_additional.py $VAR1
' >> run.sh

for ii in 98 171 175 426 570 575 589 593 669 828 829 834 837 ; do qsub -v VAR1=$ii run.sh; done


# %%


# %%



