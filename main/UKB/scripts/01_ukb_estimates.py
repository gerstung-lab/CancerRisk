'''
Builing a simple Cox model for UKB data based on AGE and SEX as a comparator to the baseline risk estimate from Denmark
- Additional AgeSex estimates 
- Compare correlation between baseline risk estimates and seperate AgeSex model.
'''

## Modules
#=======================================================================================================================

import sys
import os 
import tqdm
import h5py
import subprocess
import time
time.sleep(1)

import numpy as np 
import pandas as pd 
from sklearn import metrics
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt

from _custom_functions import KM, CIF

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Melanoma', 'Breast', 
                'Cervix Uteri', 'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney', 'Bladder', 'Brain', 'Thyroid', 'Non-Hodgkin Lymphoma', 'Multiple Myeloma', 'AML', 'Other', 'Death']

## Plotting Setup
#=======================================================================================================================

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.grid'] = False

mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['axes.titlepad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['hatch.linewidth'] = 0.07
plt.rcParams['font.size'] = 6

cm = 1/2.54
fontsize=6

colormap= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#421a01', '#6e0b3c', 
                '#9370DB', '#7A378B', '#CD6090', '#006400', '#5ebd70', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CDCB50', '#CD6600', '#FF8C69', '#8f0000', '#b3b3b3', '#454545'])

## Data Extraction
#=======================================================================================================================
time = []
pred = []
for run_id in tqdm.tqdm(range(101)):
    with h5py.File(ROOT_DIR + 'projects/CancerRisk/data/main/predictions/ukb_' + str(run_id) + '.h5', 'r') as f:
        ll = list(f.keys())
        print(len(ll))
        for ii in ll:
            time.extend(f[ii]['time'][:, :].tolist())
            pred.extend(f[ii]['pred'][:, :].tolist())
time = np.asarray(time).astype(float)
pred = np.asarray(pred).astype(float)

tt = time[:, 0]
sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)

A0 = np.load(ROOT_DIR + 'projects/CancerRisk/model/' + 'all' + '/breslow.npy')

## Prediction
#=====================================================================================================================
A0_base = []
for cc in range(22): #range(22):
    SP = []
    SP_p = []
    print(events[cc])
    if cc in [7, 8, 9, 10]:
        idx = ~sex
    elif cc in [11, 12]:
        idx = sex
    else:
        idx = np.logical_or(sex, ~sex)
    ee = pred[idx, 0, cc].copy()

    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=np.zeros_like(pred[idx, 1, :]), sex=sex.astype(int)[idx], full=False)
    pp=[]
    for ii in (range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    
    dd = pd.DataFrame(np.concatenate((tt[idx, None], ee[:, None], age[idx, None], sex[idx, None]), axis=1))
    dd.columns = ['time', 'events', 'age', 'sex']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/tmp/m.txt', sep=';')
    
    try:
        a = '''
        rm(list=ls())
        library(survival)
        ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/'

        dd <- read.csv(paste(ROOT_DIR, 'tmp/m.txt', sep=''), header=TRUE, sep=';')
        
        m <- coxph(Surv(time, events)~age+sex+I(age**2)+age:sex+I(age**2):sex, data=dd)
        
        write.csv(m$linear.predictors, '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est.csv')
        
        '''
        
        with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est.R', 'w') as write_out:
            write_out.write(a)
            
        subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est.R'], shell=False)
        
        os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est.R')
        
        est = pd.read_csv('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est.csv')
        
        os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est.csv')
        
    except:
        pass
    
    est = np.asarray(est['x'])
    A0_base.extend([est.tolist()])
    
    a,b = spearmanr(pp, est)
    SP.extend([a])
    SP_p.extend([b])
                                                                                          
    print(np.round(SP, 2))
    
    fpr, tpr, threshold = metrics.roc_curve(ee, pp)
    auc1 = metrics.auc(fpr, tpr)
    
    fpr, tpr, threshold = metrics.roc_curve(ee, est)
    auc2 = metrics.auc(fpr, tpr)
    
    dd = pd.DataFrame(np.concatenate([np.asarray(SP)[None, :].astype(float), np.asarray(SP_p)[None, :].astype(float), np.asarray([auc1])[None, :].astype(float), np.asarray([auc2])[None, :].astype(float)], axis=1))
    dd.columns = ['S_corr', 'pvalue', 'AUC_breslow', 'AUC_agesex']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/Spearman_AgeSex_Breslow.csv')
    
## PostProcessing
#=====================================================================================================================
### Save AgeSex Estimates
A0_est = np.zeros((time.shape[0], 22))
for cc in range(22): 
    if cc in [7, 8, 9, 10]:
        A0_est[~sex, cc] = np.asarray(A0_base[cc])
        A0_est[sex, cc] = 0
    elif cc in [11, 12]:
        A0_est[~sex, cc] = 0
        A0_est[sex, cc] = np.asarray(A0_base[cc])
    else:
        A0_est[:, cc] = np.asarray(A0_base[cc])
np.save(ROOT_DIR + 'projects/CancerRisk/model/' + 'all' + '/ukb_agesex.npy', A0_est)     

dd = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[0] + '/tables/Spearman_AgeSex_Breslow.csv')
for cc in range(1,22):
    dd = dd.append(pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/Spearman_AgeSex_Breslow.csv'))
dd.iloc[:, 0] = np.asarray(Events)
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/tables/Spearman_AgeSex_Breslow.csv')


print('finished')
exit()

