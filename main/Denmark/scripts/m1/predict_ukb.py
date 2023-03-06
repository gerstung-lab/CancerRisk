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
tt = [pickle.load(open(root_dir + 'main/model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]

A0 = []
for cc in tqdm.tqdm(range(22)):
    aa = pickle.load(open(root_dir + 'main/model/' + events[cc] + '/breslow.pkl', 'rb'))
    A0.extend([np.stack([[aa['female'](ii), aa['male'](ii)] for ii in range(31400)]).T])
A0 = np.stack(A0)

# %%
# Main 
# ==========================================================================================
# ==========================================================================================
for ii in [run_id]:
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    file = dir_DB + 'f_%i/f_%i/_%i' %(ii1, ii2, ii)
    file_out = dir_data + '/predictions_ukb/f_%i/f_%i/s_%i' %(ii1, ii2, ii)
    with h5py.File(file_out, 'w') as fout:
        with h5py.File(file, 'r') as f:
            idx_list = list(f.keys())
            for jj in tqdm.tqdm(samples[ii]):
                fidx = f[idx_list[jj]]
                eid = idx_list[jj]
                time, dnpr_dummies, genealogy, bth, meta = PIPE.__infer__(fidx=fidx)
                birthdate, sex, status, EOO = [meta[0], meta[1], meta[2], meta[3]]
                dates_cancer, decision, cancer_set = PIPE_cancer.__cancercall__(fidx=fidx)
                time[:, 0] = time[:, 0] - sex*100000
                time[:, 1] = time[:, 1] - sex*100000
                
                if dates_cancer:
                    if dates_cancer <= np.datetime64('2015-01-01'):
                        continue
                    
                if EOO <= np.datetime64('2015-01-01'):
                    continue

                if (np.datetime64('2015-01-01') - birthdate).astype(int) > 365*75:
                    continue

                if (np.datetime64('2015-01-01') - birthdate).astype(float) < 16*365:
                    continue
                    
                if dates_cancer != None:
                    EOO = np.minimum(EOO, dates_cancer)

                pp_main = []
                pp_add = []
                for cc in range(22):
                    pyro.clear_param_store()
                    with torch.no_grad(): 
                        mm = tt[cc]['model']
                        gg = tt[cc]['guide']
                        theta_dnpr = tt[cc]['guide'].quantiles([0.5])['theta_dnpr'][0].detach().numpy()
                        theta_gene = tt[cc]['guide'].quantiles([0.5])['theta_gene'][0].detach().numpy()
                        theta_bth = tt[cc]['guide'].quantiles([0.5])['theta_bth'][0].detach().numpy()
                        p_dnpr = np.matmul(dnpr_dummies, theta_dnpr.T)
                        p_gene = np.matmul(genealogy, theta_gene.T)
                        p_bth = np.matmul(bth, theta_bth.T)
                        pred = p_dnpr + p_gene + p_bth 
                        pp_main.append(pred)
                        pp_add.append(np.concatenate((p_dnpr, p_gene, p_bth), axis=1))
                pp_main = np.asarray(pp_main)[:, :, 0].T
                pp_add = np.asarray(pp_add)

                # Prediction
                pp_helpvar = []
                for aa in np.arange(5840, 25581, 31):
                    idx = np.logical_and(aa>=time[:, 0], aa<time[:, 1])
                    if np.any(idx):
                        pp_helpvar.extend(pp_main[idx, :].tolist())
                    else:
                        pp_helpvar.extend(np.zeros((1, 22)).tolist())
                pp_helpvar = np.asarray(pp_helpvar)       

                prediction=[]
                for cc in (range(22)):
                    cif_ = CIF(cc=cc, tt0=np.arange(5840, 25581, 31), tt_range=1825, A0=A0, pred=pp_helpvar, sex=np.repeat(sex, 637))
                    pp=[]
                    for ii in range(637):
                        pp.extend(cif_(ii))
                    pp = np.asarray(pp)
                    prediction.extend([pp[:, None]])
                prediction = np.asarray(prediction).T

                if sex==0:
                    prediction[:, :, 11:13] = 0
                else:
                    prediction[:, :, 7:11] = 0

                # observation period from assessment to EOO
                idx_obs = np.logical_and(np.arange(5840, 25581, 31) >= np.min(time[:, 0]), np.arange(5840, 25581, 31) <= (np.minimum(EOO, np.datetime64('2015-01-01')) - birthdate).astype(float))
                prediction[:, ~idx_obs, :] = 0
                

                if np.any(decision != None): 
                    decision = np.concatenate((decision, np.zeros((1,)).astype(float)))
                else:
                    decision = np.zeros((21, )).astype(float)
                    decision = np.concatenate((decision, (meta[2]==90).astype(float)))

                
                #TimeToEnd, Sex, Age, decision, pred
                res1 = np.concatenate((
                (EOO - np.datetime64('2015-01-01')).astype(float), sex, (np.datetime64('2015-01-01')-birthdate).astype(float) ))[None, :]
                res2 = np.stack((decision[None, :], pp_main[-1, :][None, :]), axis=1)

                fout.create_group(str(eid))
                fout[str(eid)].create_dataset('time', data=res1.astype(float), maxshape=(1, 3), compression='lzf')
                fout[str(eid)].create_dataset('pred', data=res2.astype(float), maxshape=(1, 2, 22), compression='lzf')
                fout[str(eid)].create_dataset('pred_sub', data=np.asarray(pp_add)[:, -1, :].astype(float), maxshape=(22, 3), compression='lzf')

                fout[str(eid)].create_dataset('absolute_risk', data=prediction.astype(float), maxshape=(1, 637, 22), compression='lzf')
                #fout[str(eid)].create_dataset('dyn_time', data=time[:, :2].astype(float), maxshape=(None, 2), compression='lzf')
                #fout[str(eid)].create_dataset('dyn_decision', data=decision.astype(float), maxshape=(22,), compression='lzf')
                #fout[str(eid)].create_dataset('dyn_pred', data=pp_main.astype(float), maxshape=(None, 22), compression='lzf')


# %%
exit()

# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N predict
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=3gb
#PBS -l walltime=25:00:00
#PBS -t 0-200

cd $PBS_O_WORDIR
module load anaconda3/2019.10
source conda activate

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/m1/predict_ukb.ipynb --output /users/projects/cancer_risk/main/scripts/m1/predict_ukb

/services/tools/anaconda3/2019.10/bin/python3.7 /users/projects/cancer_risk/main/scripts/m1/predict_ukb.py $PBS_ARRAYID
' >> run.sh

qsub run.sh


# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N predict
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=3gb
#PBS -l walltime=5:00:00

cd $PBS_O_WORDIR
module load anaconda3/2019.10
source conda activate

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/m1/predict_ukb.ipynb --output /users/projects/cancer_risk/main/scripts/m1/predict_ukb

/services/tools/anaconda3/2019.10/bin/python3.7 /users/projects/cancer_risk/main/scripts/m1/predict_ukb.py $VAR1
' >> run.sh

for ii in 717 722 727 732 737 742 747 752 757 889 892 ; do qsub -v VAR1=$ii run.sh; done


# %%
##### 

# %%
'''
# Combine 
# ==========================================================================================
# ==========================================================================================
file_out = dir_data + '/predictions_ukb/master.h5'
with h5py.File(file_out, 'w') as fout:
    fout.create_dataset("data", (4248491, 637, 22), dtype='float')
    n = 0
    for ii in tqdm.tqdm(range(1000)):
        ii1 = ii//100
        ii2 = (ii - ii1*100)//10
        file = dir_data + '/predictions_ukb/f_%i/f_%i/s_%i' %(ii1, ii2, ii)
        with h5py.File(file, 'r') as f:
            idx_list = list(f.keys())
            for eid in idx_list: 
                fout['data'][n, :, :] = f[str(eid)]['absolute_risk'][:, :, :].astype(float)
                n += 1
'''


