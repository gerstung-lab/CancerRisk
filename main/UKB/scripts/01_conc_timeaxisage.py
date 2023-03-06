'''
Estimate Concordance index for predictions when timeline is age
'''


## Custom Modules
#=======================================================================================================================
import sys
import os 
import tqdm
import h5py
import dill
import subprocess
import time
from _custom_functions import KM, CIF, metric_table

import numpy as np 
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
                'Cervix Uteri', 'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney', 'Bladder', 'Brain',
                'Thyroid', 'Non-Hodgkin Lymphoma', 'Multiple Myeloma', 'AML', 'Other', 'Death']


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

colormap= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#421a01', '#6e0b3c', 
                '#9370DB', '#7A378B', '#CD6090', '#006400', '#5ebd70', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CDCB50', '#CD6600', '#FF8C69', '#8f0000', '#b3b3b3', '#454545'])


## Extract Data
#=======================================================================================================================
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

A0 = np.load(ROOT_DIR + 'projects/CancerRisk/model/' + 'all' + '/breslow.npy')

## Predictions + Plot
#=======================================================================================================================
concordance=[]
for cc in tqdm.tqdm(range(22)):
    dd = np.concatenate((age[:, None], (age+tt)[:, None], pred[:, 0, cc, None], pred[:, 1, cc, None], sex[:, None]),  axis=1).astype(float)
    dd = pd.DataFrame(dd)
    dd.columns = ['start', 'stop', 'events', 'pred', 'sex']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/concordance_raw.csv', sep=';')
    
    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/concordance_raw.csv'"

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

    write(paste(x_f, x_m, x, sep=';'), file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc2.txt', sep=''), append=FALSE, sep=";")
    '''

    with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.R'], shell=False)

    concordance.extend(np.asarray(pd.read_csv('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc2.txt', sep=';', header=None)).tolist())
    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.R')                      
    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc2.txt')

dd = pd.DataFrame(concordance)
dd.columns = ['concordance_f', 'se_f', 'concordance_m', 'se_m', 'concordance', 'se']
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/concordance2.csv', sep=';')

print('finished')
exit()
