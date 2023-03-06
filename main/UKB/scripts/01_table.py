'''
Extract some baseline factors from UKB data form overrview Table
'''

## Custom Modules
#=======================================================================================================================
import sys
import os 
import tqdm
import h5py
import dill
import subprocess

import numpy as np 
np.set_printoptions(suppress=True)
import pandas as pd 
from sklearn import metrics
from scipy import interp
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'


events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Melanoma', 'Breast', 
                'Cervix Uteri', 'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney', 'Bladder', 'Brain'
                'Thyroid', 'Non-Hodgkin Lymphoma', 'Multiple Myeloma', 'AML', 'Other', 'Death']

## Custom Functions
#=======================================================================================================================
def IQR(x, idx=None):
    if np.any(idx==None):
        a,b,c = np.round(np.quantile(x, [0.05, 0.5, 0.95]), 4)
    else:
        a,b,c = np.round(np.quantile(x[idx], [0.05, 0.5, 0.95]), 4)
    return(['[ ' + str(a) + ' - ' + str(b) + ' - ' + str(c) + ' ]'])


## Extract Data
#=======================================================================================================================
res = np.zeros((1, 7))
for run_id in tqdm.tqdm(range(101)):
    res_ = np.load(ROOT_DIR + 'projects/CancerRisk/data/main/table/res_' + str(run_id) + '.npy')[None, :]
    res = np.concatenate((res, res_))
res.sum(axis=0)
   
unique_diagnoses = []
lifeyears = []
family_info = []
visits = []
mother = []
decision = []
genealogy = []
baseline = []

for run_id in tqdm.tqdm(range(101)):
    with h5py.File(ROOT_DIR + 'projects/CancerRisk/data/main/table/ukb_' + str(run_id) + '.h5', 'r') as f:
        ll = list(f.keys())
        for ii in ll:
            unique_diagnoses.extend(f[ii]['unique_diagnoses'][:].tolist())
            lifeyears.extend(f[ii]['lifeyears'][:].tolist())
            family_info.extend(f[ii]['family_info'][:].tolist())
            visits.extend(f[ii]['visits'][:].tolist())
            mother.extend(f[ii]['mother'][:].tolist())
            decision.extend(f[ii]['decision'][:][None, :].tolist())
            genealogy.extend(f[ii]['genealogy'][:, :].tolist())
            baseline.extend(f[ii]['baseline'][:, :].tolist())

time = []
pred = []
for run_id in tqdm.tqdm(range(101)):
    with h5py.File(ROOT_DIR + 'projects/CancerRisk/data/main/predictions/ukb_' + str(run_id) + '.h5', 'r') as f:
        ll = list(f.keys())
        for ii in ll:
            time.extend(f[ii]['time'][:, :].tolist())
            pred.extend(f[ii]['pred'][:, :].tolist())
time = np.asarray(time).astype(float)
pred = np.asarray(pred).astype(float)
tt = time[:, 0]
sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)



## Table
#=======================================================================================================================

unique_diagnoses = np.asarray(unique_diagnoses)
lifeyears = np.asarray(lifeyears)
family_info = np.asarray(family_info)
visits = np.asarray(visits)
mother = np.asarray(mother)
decision = np.asarray(decision)
genealogy = np.asarray(genealogy)
baseline = np.asarray(baseline)
    
family_info.sum()
    
(sex==0).sum()
(sex==1).sum()

unique_diagnoses.sum()
never_shows = unique_diagnoses==0
never_shows.sum()
IQR(unique_diagnoses, idx=~never_shows)
lifeyears.sum()
IQR(lifeyears)
visits.sum()
IQR(visits)


print('alc')
(baseline[:, 0]==1).sum()
(baseline[:, 0]==-1).sum()

print('smok')
(baseline[:, 1]==1).sum()
(baseline[:, 1]==-1).sum()

print('highBP')
(baseline[:, 2]==1).sum()
(baseline[:, 2]==-1).sum()

print('lowBP')
(baseline[:, 3]==1).sum()
(baseline[:, 3]==-1).sum()


IQR(baseline[:, 4]*100+170)

IQR(baseline[:, 5]*100+75)

IQR(baseline[:, 6]*100, idx=baseline[:, 6]>0)


np.logical_and(baseline[:, 6]>0, sex==0).sum()


np.round(decision[sex==0].sum(axis=0))
np.round(decision[sex==1].sum(axis=0))


(genealogy >0).sum(axis=0)
