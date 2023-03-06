# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import dill as pickle
import tqdm
import subprocess

import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

import torch 
import pyro

# Custom Functions
sys.path.append('/users/projects/cancer_risk/main/scripts/ProbCox')
from _custom_functions import KM, CIF, metric_table, round_

# seeding
np.random.seed(11)
torch.random.manual_seed(12)
pyro.set_rng_seed(13)

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'
dir_out = '/users/projects/cancer_risk/main/output/'
dir_pred = '/users/projects/cancer_risk/data/predictions_ukb/'
dir_root = '/users/projects/cancer_risk/'

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver',
                'Pancreas', 'Lung', 'Melanoma', 'Breast', 'Cervix uteri',
                'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney',
                'Bladder', 'Brain', 'Thyroid', 'NHL', 'MM', 'AML', 'Other', 'Death']

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

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
mpl.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['font.size'] = 6
fontsize=6
cm = 1/2.54

colormap= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#421a01', '#6e0b3c', 
                '#9370DB', '#7A378B', '#CD6090', '#006400', '#5ebd70', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CDCB50', '#CD6600', '#FF8C69', '#8f0000', '#b3b3b3', '#454545'])
cmap = mpl.cm.get_cmap('RdBu')

# %%
# Models 
# ==========================================================================================
# ==========================================================================================
tt = [pickle.load(open(dir_root + 'main/model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]

A0 = []
for cc in (range(22)):
    aa = pickle.load(open(dir_root + 'main/model/' + events[cc] + '/breslow.pkl', 'rb'))
    A0.extend([np.stack([[aa['female'](ii), aa['male'](ii)]for ii in range(31400)]).T])
A0 = np.stack(A0)

# %%
## Extract Data
#=======================================================================================================================
time = []
pred = []

file = dir_data + 'predictions_ukb/master_quick.h5'
with h5py.File(file, 'r') as f:
    time = f['time'][:]
    pred = f['pred'][:]

tt_surv = time[:, 0] + 1
sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)
out = pred[:, 0, :].copy()

idxage = np.logical_and(age/365>=50, age/365<=75)
tt_surv = tt_surv[idxage]
sex =sex[idxage]
age = age[idxage]
out =out[idxage]

time = time[idxage]
pred = pred[idxage]
pred2 = pred2[idxage]

N = time.shape[0]


# %%
## Time-Dependent Concordance Evaluation
#=======================================================================================================================
concordance=[]
for cc in tqdm.tqdm(range(22)):
    dd = np.concatenate((age[:, None], (age+tt_surv)[:, None], pred[:, 0, cc, None], pred[:, 1, cc, None], sex[:, None]),  axis=1).astype(float)
    dd = pd.DataFrame(dd)
    dd.columns = ['start', 'stop', 'events', 'pred', 'sex']
    dd.to_csv(dir_out + events[cc] + '/data/concordance_ukb_raw.csv', sep=';')
    
    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/users/projects/cancer_risk/'
    '''

    b = 'data_name = ' + "'main/output/" + str(events[cc]) + "/data/concordance_ukb_raw.csv'"

    c = '''
    dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')

    m = coxph(Surv(start, stop, events)~pred, data=dd[dd$sex==0, ])
    x_f = paste(unname(summary(m)$concordance[1]), unname(summary(m)$concordance[2]), sep=';')

    m = coxph(Surv(start, stop, events)~pred, data=dd[dd$sex==1, ])
    x_m = paste(unname(summary(m)$concordance[1]), unname(summary(m)$concordance[2]), sep=';')

    dd$start = dd$start + dd$sex * 1000000
    dd$stop = dd$stop + dd$sex * 1000000
    m = coxph(Surv(start, stop, events)~pred, data=dd)
    x = paste(unname(summary(m)$concordance[1]), unname(summary(m)$concordance[2]), sep=';')

    write(paste(x_f, x_m, x, sep=';'), file=paste('/users/projects/cancer_risk/tmp/est_conc2.txt', sep=''), append=FALSE, sep=";")
    '''

    with open('/users/projects/cancer_risk/tmp/est_conc.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/users/projects/cancer_risk/tmp/est_conc.R'], shell=False)

    concordance.extend(np.asarray(pd.read_csv('/users/projects/cancer_risk/tmp/est_conc2.txt', sep=';', header=None)).tolist())
    os.remove('/users/projects/cancer_risk/tmp/est_conc.R')                      
    os.remove('/users/projects/cancer_risk/tmp/est_conc2.txt')

dd = pd.DataFrame(concordance)
dd.columns = ['concordance_f', 'se_f', 'concordance_m', 'se_m', 'concordance', 'se']
dd.to_csv(dir_out + 'main' + '/data/concordance2_ukb.csv', sep=';')

# %%
dd

# %%
## Metrics / Calibration
#=======================================================================================================================
concordance=[]
for cc in tqdm.tqdm(range(22)):
    print(events[cc])
    if cc in [7, 8, 9, 10]:
        idx = ~sex
    elif cc in [11, 12]:
        idx = sex
    else:
        idx = np.logical_or(sex, ~sex)
        
    ee = pred[idx, 0, cc].copy()
    y_ = ee
    tt_ = tt_surv[idx].copy()
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=pred[idx, 1, :], sex=sex.astype(int)[idx])
    pp=[]
    for ii in (range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    
    dd = pd.DataFrame(np.concatenate((tt_surv[idx, None], y_[:, None], pp[:, None]), axis=1).astype(float))
    dd.columns = ['time', 'events', 'split']
    dd.to_csv(dir_out + events[cc] + '/data/metrics_ukb_raw.csv', sep=';')
    
    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/users/projects/cancer_risk/main/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/metrics_ukb_raw.csv'"

    c = '''
    dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')

    m = coxph(Surv(time, events)~split, data=dd)

    x = paste(unname(summary(m)$concordance[1]), unname(summary(m)$concordance[2]), sep=';')

    write(x, file=paste('/users/projects/cancer_risk/tmp/est_conc.txt', sep=''), append=FALSE, sep=";")
    '''

    with open('/users/projects/cancer_risk/tmp/est_conc.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/users/projects/cancer_risk/tmp/est_conc.R'], shell=False)
    os.remove('/users/projects/cancer_risk/tmp/est_conc.R')

    conc = np.squeeze(np.loadtxt('/users/projects/cancer_risk/tmp/est_conc.txt', delimiter=';'))
    concordance.extend([conc.tolist()])

dd = pd.DataFrame(concordance)
dd.columns = ['concordance', 'se']
dd.to_csv(dir_out + 'main' + '/data/concordance_ukb.csv', sep=';')

# %%
dd

# %%
print('finished')
exit()

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



