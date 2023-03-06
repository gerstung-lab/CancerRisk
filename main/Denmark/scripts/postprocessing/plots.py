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
pred2 = []

file = dir_data + 'predictions_ukb/master_quick.h5'
with h5py.File(file, 'r') as f:
    time = f['time'][:]
    pred = f['pred'][:]
    pred2 = f['pred2'][:]

tt_surv = time[:, 0] + 1
sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)
out = pred[:, 0, :].copy()
N = time.shape[0]


# %%
## Time-Dependent Concordance Evaluation
#=======================================================================================================================
concordance=[]
for cc in tqdm.tqdm(range(22)):
    dd = np.concatenate((age[:, None], (age+tt_surv)[:, None], pred[:, 0, cc, None], pred[:, 1, cc, None], sex[:, None]),  axis=1).astype(float)
    dd = pd.DataFrame(dd)
    dd.columns = ['start', 'stop', 'events', 'pred', 'sex']
    dd.to_csv(dir_out + events[cc] + '/data/concordance_raw.csv', sep=';')
    
    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/users/projects/cancer_risk/'
    '''

    b = 'data_name = ' + "'main/output/" + str(events[cc]) + "/data/concordance_raw.csv'"

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
dd.to_csv(dir_out + 'main' + '/data/concordance2.csv', sep=';')

# %%
## Proportional Hazard Test
#======================================================================================================================
a = '''
rm(list=ls())
library(survival)
library(survminer)

ROOT_DIR = '/users/projects/cancer_risk/'

events = c('oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
            'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
            'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death')
res=c()
for(cc in seq(1, 22)){
print(events[cc])
dd <- read.csv(paste(ROOT_DIR, 'main/output/', events[cc], '/data/concordance_raw.csv', sep=''), header=TRUE, sep=';')
dd$start = dd$start + dd$sex * 1000000
dd$stop = dd$stop + dd$sex * 1000000
m = coxph(Surv(start, stop, events)~pred, data=dd)

data_folder = paste('/users/projects/cancer_risk/main/output/', events[cc], '/', sep='')

test.ph <- cox.zph(m)
write.csv(test.ph$table, paste(data_folder, 'tables/prophaz.csv', sep=''))
res = c(res, matrix(test.ph$table[1, ]))
}

write.csv(data.frame(matrix(res, ncol=3,  byrow=TRUE)), '/users/projects/cancer_risk/main/output/main/tables/prophaz.csv')
'''

with open('/users/projects/cancer_risk/tmp/test.R', 'w') as write_out:
    write_out.write(a)

subprocess.check_call(['Rscript', '/users/projects/cancer_risk/tmp/test.R'], shell=False)
os.remove('/users/projects/cancer_risk/tmp/test.R')

# %%
## Attribution 
#======================================================================================================================
prop_dnpr1=[]
prop_gene1=[]
prop_bth1=[]

prop_dnpr2=[]
prop_gene2=[]
prop_bth2=[]

prop_dnpr3=[]
prop_gene3=[]
prop_bth3=[]

for cc in tqdm.tqdm(range(22)):
    idx = age <= 45*365
    pabs = np.abs(pred2[idx, cc, :]).sum(axis=-1)
    prop_dnpr1.extend([(np.abs(pred2[idx, cc, 0])/pabs).mean()])
    prop_gene1.extend([(np.abs(pred2[idx, cc, 1])/pabs).mean()])
    prop_bth1.extend([(np.abs(pred2[idx, cc, 2])/pabs).mean()])
    
    idx = np.logical_and(age > 45*365, age <= 65*365)
    pabs = np.abs(pred2[idx, cc, :]).sum(axis=-1)
    prop_dnpr2.extend([(np.abs(pred2[idx, cc, 0])/pabs).mean()])
    prop_gene2.extend([(np.abs(pred2[idx, cc, 1])/pabs).mean()])
    prop_bth2.extend([(np.abs(pred2[idx, cc, 2])/pabs).mean()])
    
    idx = age > 65*365
    pabs = np.abs(pred2[idx, cc, :]).sum(axis=-1)
    prop_dnpr3.extend([(np.abs(pred2[idx, cc, 0])/pabs).mean()])
    prop_gene3.extend([(np.abs(pred2[idx, cc, 1])/pabs).mean()])
    prop_bth3.extend([(np.abs(pred2[idx, cc, 2])/pabs).mean()])
    
prop_dnpr1 = np.asarray(prop_dnpr1)
prop_gene1 = np.asarray(prop_gene1)
prop_bth1 = np.asarray(prop_bth1)
prop_dnpr2 = np.asarray(prop_dnpr2)
prop_gene2 = np.asarray(prop_gene2)
prop_bth2 = np.asarray(prop_bth2) 
prop_dnpr3 = np.asarray(prop_dnpr3)
prop_gene3 = np.asarray(prop_gene3)
prop_bth3 = np.asarray(prop_bth3) 

fig, ax = plt.subplots(1, 1, figsize=(17*cm,2*cm), dpi=600)

ax.bar(np.arange(22)-0.3, prop_dnpr1, width=0.27, lw=0.05, color=colormap, edgecolor=colormap, hatch='')
ax.bar(np.arange(22), prop_dnpr2, width=0.27, lw=0.05, color=colormap, edgecolor=colormap, hatch='')
ax.bar(np.arange(22)+0.3, prop_dnpr3, width=0.27, lw=0.05, color=colormap, edgecolor=colormap, hatch='', label='DNPR')

ax.bar(np.arange(22)-0.3, prop_bth1, width=0.27, bottom=prop_dnpr1, lw=0.05, color='white', edgecolor=colormap, hatch='xxxxxxxxxx')
ax.bar(np.arange(22), prop_bth2, width=0.27, bottom=prop_dnpr2, lw=0.05, color='white', edgecolor=colormap, hatch='xxxxxxxxxx')
ax.bar(np.arange(22)+0.3, prop_bth3, width=0.27, bottom=prop_dnpr3, lw=0.05, color='white', edgecolor=colormap, hatch='xxxxxxxxxx', label='BTH')

ax.bar(np.arange(22)-0.3, prop_gene1, width=0.27, color=colormap, bottom=prop_dnpr1+prop_bth1, lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx')
ax.bar(np.arange(22), prop_gene2, width=0.27, color=colormap, bottom=prop_dnpr2+prop_bth2, lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx')
ax.bar(np.arange(22)+0.3, prop_gene3, width=0.27, color=colormap, bottom=prop_dnpr3+prop_bth3, lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx', label='Genealogy')

ax.set_xlim([-0.75, 21.75])
ax.set_xticks(np.arange(22))
ax.set_xticklabels(np.asarray(Events), rotation=90)

handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[-1][-1], handles[-2][-1], handles[-3][-1]], labels[::-1], frameon=False, fontsize=5, bbox_to_anchor=(1.0, 0.75))

plt.savefig(dir_out + 'main' + '/figures/attribution.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out + 'main' + '/figures/attribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   

    
dd = pd.DataFrame(np.concatenate((np.zeros((9, 1)),
    np.concatenate((prop_dnpr1[None, :],
prop_gene1[None, :],
prop_bth1[None, :],
prop_dnpr2[None, :],
prop_gene2[None, :],
prop_bth2[None, :],
prop_dnpr3[None, :],
prop_gene3[None, :],
prop_bth3[None, :]), axis=0)), axis=1))
dd.iloc[:, 0] = ['prop_dnpr_45', 'prop_gene_45', 'prop_bth_45',
                 'prop_dnpr_45_65', 'prop_gene_45_65', 'prop_bth_45_65',
                 'prop_dnpr_65', 'prop_gene_65', 'prop_bth_65']
names = ['']
names.extend(events)
dd.columns = names
dd.to_csv(dir_out + 'main' + '/data/attribution_comparision.csv')


# version 2
# plot 6 
def IQR(x):
    try:
        a, b = np.quantile(x, [0.025, 0.975])
    except:
        a, b = 0, 0 
    return(b-a)

prop_dnpr1=[]
prop_gene1=[]
prop_bth1=[]
prop1=[]

prop_dnpr2=[]
prop_gene2=[]
prop_bth2=[]
prop2=[]

prop_dnpr3=[]
prop_gene3=[]
prop_bth3=[]
prop3=[]

for cc in range(22):
    idx = age <= 45*365
    prop_dnpr1.extend([(IQR(pred2[idx, cc, 0]))])
    prop_gene1.extend([(IQR(pred2[idx, cc, 1]))])
    prop_bth1.extend([(IQR(pred2[idx, cc, 2]))])
    prop1.extend([(IQR(pred2[idx, cc, 0] + pred2[idx, cc, 1] + pred2[idx, cc, 2]))])
    
    idx = np.logical_and(age > 45*365, age <= 65*365)
    pabs = np.abs(pred2[idx, cc, :]).sum(axis=-1)
    prop_dnpr2.extend([(IQR(pred2[idx, cc, 0]))])
    prop_gene2.extend([(IQR(pred2[idx, cc, 1]))])
    prop_bth2.extend([(IQR(pred2[idx, cc, 2]))])
    prop2.extend([(IQR(pred2[idx, cc, 0] + pred2[idx, cc, 1] + pred2[idx, cc, 2]))])
    
    idx = age > 65*365
    pabs = np.abs(pred2[idx, cc, :]).sum(axis=-1)
    prop_dnpr3.extend([(IQR(pred2[idx, cc, 0]))])
    prop_gene3.extend([(IQR(pred2[idx, cc, 1]))])
    prop_bth3.extend([(IQR(pred2[idx, cc, 2]))])
    prop3.extend([(IQR(pred2[idx, cc, 0] + pred2[idx, cc, 1] + pred2[idx, cc, 2]))])
        

prop_dnpr1 = np.asarray(prop_dnpr1)
prop_gene1 = np.asarray(prop_gene1)
prop_bth1 = np.asarray(prop_bth1)
prop1 = np.asarray(prop1)
prop_dnpr2 = np.asarray(prop_dnpr2)
prop_gene2 = np.asarray(prop_gene2)
prop_bth2 = np.asarray(prop_bth2) 
prop2 = np.asarray(prop2) 
prop_dnpr3 = np.asarray(prop_dnpr3)
prop_gene3 = np.asarray(prop_gene3)
prop_bth3 = np.asarray(prop_bth3) 
prop3 = np.asarray(prop3) 


fig, ax = plt.subplots(1, 1, figsize=(17*cm,2*cm), dpi=600)

ax.bar(np.arange(22)-0.3, prop_dnpr1, width=0.27, lw=0.05, color=colormap, edgecolor=colormap, hatch='')
ax.bar(np.arange(22), prop_dnpr2, width=0.27, lw=0.05, color=colormap, edgecolor=colormap, hatch='')
ax.bar(np.arange(22)+0.3, prop_dnpr3, width=0.27, lw=0.05, color=colormap, edgecolor=colormap, hatch='', label='DNPR')

ax.bar(np.arange(22)-0.3, prop_bth1, width=0.27, bottom=prop_dnpr1, lw=0.05, color='white', edgecolor=colormap, hatch='xxxxxxxxxx')
ax.bar(np.arange(22), prop_bth2, width=0.27, bottom=prop_dnpr2, lw=0.05, color='white', edgecolor=colormap, hatch='xxxxxxxxxx')
ax.bar(np.arange(22)+0.3, prop_bth3, width=0.27, bottom=prop_dnpr3, lw=0.05, color='white', edgecolor=colormap, hatch='xxxxxxxxxx', label='BTH')

ax.bar(np.arange(22)-0.3, prop_gene1, width=0.27, color=colormap, bottom=prop_dnpr1+prop_bth1, lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx')
ax.bar(np.arange(22), prop_gene2, width=0.27, color=colormap, bottom=prop_dnpr2+prop_bth2, lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx')
ax.bar(np.arange(22)+0.3, prop_gene3, width=0.27, color=colormap, bottom=prop_dnpr3+prop_bth3, lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx', label='Genealogy')

ax.plot(np.arange(22)-0.3, prop1, ls='', marker='_', markersize=2, color='black')
ax.plot(np.arange(22), prop2, ls='', marker='_', markersize=2, color='black')
ax.plot(np.arange(22)+0.3, prop3, ls='', marker='_', markersize=2, color='black')

#ax.set_yticklabels(['0', '0.5', '1])
ax.set_xlim([-0.75, 21.75])
ax.set_xticks(np.arange(22))
ax.set_xticklabels(np.asarray(Events), rotation=90)

handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[-1][-1], handles[-2][-1], handles[-3][-1]], labels[::-1], frameon=False, fontsize=5, bbox_to_anchor=(1.0, 0.75))

plt.savefig(dir_out + 'main' + '/figures/attribution_v2.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out + 'main' + '/figures/attribution_v2.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   

dd = pd.DataFrame(np.concatenate((np.zeros((12, 1)),
    np.concatenate((prop_dnpr1[None, :],
prop_gene1[None, :],
prop_bth1[None, :],
prop1[None, :],
prop_dnpr2[None, :],
prop_gene2[None, :],
prop_bth2[None, :],
prop2[None, :],
prop_dnpr3[None, :],
prop_gene3[None, :],
prop_bth3[None, :],
prop3[None, :]), axis=0)), axis=1))
dd.iloc[:, 0] = ['prop_dnpr_45', 'prop_gene_45', 'prop_bth_45', 'prop_45',
                 'prop_dnpr_45_65', 'prop_gene_45_65', 'prop_bth_45_65', 'prop_45_65',
                 'prop_dnpr_65', 'prop_gene_65', 'prop_bth_65', 'prop_65']
names = ['']
names.extend(events)
dd.columns = names
dd.to_csv(dir_out + 'main' + '/data/attribution_comparision_v2.csv')






# %%
## Correlation
#======================================================================================================================

full = pred[:, 1, :20]
genealogy = pred2[:, :20, 1]
health = pred2[:, :20, 0] + pred2[:, :20, 2]
idxevents = pred[:, 0, :20].max(axis=1)==1

def corplot(dd, suffix='', sex=sex, idxevents=idxevents, sizemark=28, cmap=cmap, Events=Events, ROOT_DIR=dir_out, subset=True, corr=pearsonr):
    mat = np.zeros((20, 20))
    mat_p = np.zeros((20, 20))
    # female
    for ii in tqdm.tqdm(np.arange(20)):
        if ii not in [11, 12]:
            for jj in np.arange(ii, 20):
                if jj not in [11, 12]:
                    idx = np.logical_and(~sex, idxevents)
                    if subset:
                        idx = np.logical_and(idx, np.any((dd != 0), axis=1))
                    mat[ii, jj], mat_p[ii, jj] = corr(dd[idx, ii], dd[idx, jj])    
    print(idx.sum())
    # male
    for ii in tqdm.tqdm(np.arange(20)):
        if ii not in [7, 8, 9, 10]:
            for jj in np.arange(ii, 20):
                if jj not in [7, 8, 9, 10]:
                    idx = np.logical_and(sex, idxevents)
                    if subset:
                        idx = np.logical_and(idx, np.any((dd != 0), axis=1))
                    mat[jj, ii], mat_p[jj, ii] = corr(dd[idx, jj], dd[idx, ii]) 
    print(idx.sum())
                      
    mat_pcor = multipletests(mat_p.reshape(400), alpha=0.1, method='hommel', is_sorted=False, returnsorted=False)[1].reshape(20, 20)

    out = pd.DataFrame(np.concatenate((np.zeros((20, 1)), mat, mat_pcor, mat_p), axis=1))
    out.iloc[:, 0] = Events[:20]
    out.to_csv(dir_out  + 'main' + '/tables/corr_' + suffix + '.csv', sep=';')

    fig, ax = plt.subplots(1, 1, figsize=(8*cm,8*cm), dpi=300)
    for ii in range(20):
        for jj in range(20):
            ax.scatter(x=ii, y=jj, s=sizemark*np.abs(mat[ii, jj]), marker='s', color=cmap(0.5 - mat[ii, jj]/2))
    ax.scatter(x=range(20), y=range(20), s=sizemark, marker='s', color='black') 
    ax.set_xticks(np.arange(-1, 20))
    ax.set_xticklabels(['', 'Oesophagus',
     'Stomach',
     'Colorectal',
     'Liver',
     'Pancreas',
     'Lung',
     'Melanoma',
     'Breast',
     'Cervix Uteri',
     'Corpus Uteri',
     'Ovary',
     'Prostate',
     'Testis',
     'Kidney',
     'Bladder',
     'Brain',
     'Thyroid',
     'NHL',
     'MM',
     'AML'], rotation=90)
    ax.set_yticks(range(20))
    ax.set_yticklabels(np.asarray(Events)[:20], rotation=0)
    ax.set_xlim([-0.5, 19.5])
    ax.set_ylim([-0.5, 19.5])
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    plt.savefig(dir_out +  'main' + '/figures/corrplot_' + suffix + '.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out +  'main' + '/figures/corrplot_' + suffix + '.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()  

# Main Figure
corplot(dd=full, suffix='full_spearman', corr=pearsonr)
corplot(dd=genealogy, suffix='genealogy_spearman', corr=pearsonr)
corplot(dd=health, suffix='health_spearman', corr=pearsonr)

corplot(dd=full, suffix='full_pearson', corr=pearsonr)
corplot(dd=genealogy, suffix='genealogy_pearson', corr=pearsonr)
corplot(dd=health, suffix='health_pearson', corr=pearsonr)

# %%
## Kaplan-Meier Curves
#======================================================================================================================
estll = []
for cc in tqdm.tqdm(range(0, 22)):
    print(events[cc])
    if cc in [7, 8, 9, 10]:
        idx = ~sex
    elif cc in [11, 12]:
        idx = sex
    else:
        idx = np.logical_or(sex, ~sex)
        
    ee = pred[idx, 0, cc].copy()
    y_ = ee
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=pred[idx, 1, :], sex=sex.astype(int)[idx], full=False)
    pp=[]
    for ii in (range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    lr1 = pp < np.quantile(pp, 0.99)
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=np.zeros_like(pred[idx, 1, :]), sex=sex.astype(int)[idx], full=False)
    pp=[]
    for ii in (range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    lr2 = pp < np.quantile(pp, 0.99)
    
    tt_ = np.concatenate((tt_surv[idx][~lr1], tt_surv[idx][~lr2]))
    ee = np.concatenate((ee[~lr1], ee[~lr2]))
    lr = np.concatenate((np.zeros(((~lr1).sum())), np.ones(((~lr2).sum())))).astype(bool)
    
    dd = pd.DataFrame(np.concatenate((tt_[:, None], ee[:, None], (~lr[:, None]).astype(float)), axis=1))
    dd.columns = ['time', 'events', 'split']
    dd.to_csv(dir_out + events[cc] + '/data/kmplots_raw.csv', sep=';')

    try:
        a = '''
        rm(list=ls())
        library(survival)
        ROOT_DIR = '/users/projects/cancer_risk/main/'
        '''

        b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/kmplots_raw.csv'"

        c = '''
        dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')

        m = coxph(Surv(time, events)~split, data=dd)

        x = paste(unname(exp(m$coefficients)), unname(exp(m$coefficients - 1.96*sqrt(m$var))), unname(exp(m$coefficients + 1.96*sqrt(m$var))), sep=';')

        write(x, file=paste('/users/projects/cancer_risk/tmp/est_surv.txt', sep=''), append=FALSE, sep=";")

        surv_diff <- survdiff(Surv(time, events) ~ split, data = dd)
        x = paste(surv_diff$chisq, pchisq(surv_diff$chisq, 1, lower.tail=FALSE), sep=';')
        write(x, file=paste('/users/projects/cancer_risk/tmp/est_logrank.txt', sep=''), append=FALSE, sep=";")

        '''

        with open('/users/projects/cancer_risk/tmp/est_surv.R', 'w') as write_out:
            write_out.write(a+b+c)

        subprocess.check_call(['Rscript', '/users/projects/cancer_risk/tmp/est_surv.R'], shell=False)
        os.remove('/users/projects/cancer_risk/tmp/est_surv.R')

        hr_est = np.loadtxt('/users/projects/cancer_risk/tmp/est_surv.txt', delimiter=';')
        os.remove('/users/projects/cancer_risk/tmp/est_surv.txt')

        logr_est = np.loadtxt('/users/projects/cancer_risk/tmp/est_logrank.txt', delimiter=';')
        os.remove('/users/projects/cancer_risk/tmp/est_logrank.txt')

        if np.abs(hr_est[0]) < 100000:
            hr_text = 'HR: ' + str(np.asarray(np.round(hr_est[0], 2))) + ' (' + str(np.asarray(np.round(hr_est[1], 2))) + ', ' + str(np.asarray(np.round(hr_est[2], 2))) + ')'
        else:
            hr_text = 'HR: -'
            
        if np.abs(hr_est[0]) < 100000:
            if np.logical_or(np.logical_and(hr_est[1] > 1, hr_est[2] > 1), np.logical_and(hr_est[1] < 1, hr_est[2] < 1)):
                hr_text_small = '' + str(np.asarray(np.round(hr_est[0], 2))) + '*'
            else:
                hr_text_small = '' + str(np.asarray(np.round(hr_est[0], 2))) + ''
        else:
            hr_text_small = '-'

    except:
            hr_text = 'HR: -'
            hr_text_small = '-'
            logr_est = ['', '', '',  '']
            print('error')

    estll.extend([np.concatenate((hr_est, logr_est)).tolist()])

    fig, ax = plt.subplots(2, 1, figsize=(3.75*cm,4*cm), dpi=600, gridspec_kw={'height_ratios':[4, 1]})
    fig.subplots_adjust(wspace=0.0, hspace=0.4)

    times, km, ci, ll_1 = KM(tt_[lr], ee[lr], t1=[0, 365, 730, 1095], tmax=1196)
    m1 = np.min(ci[0])
    dd = pd.DataFrame(np.concatenate((times[:, None], km[:, None], ci[0][:, None], ci[1][:, None]), axis=1))
    dd.columns = ['times', 'km', 'lower95', 'upper95']
    dd.to_csv(dir_out +  events[cc] + '/data/kmplots1.csv', sep=';')

    ax[0].step(times, km, where='post', color='black',  lw=1)
    ax[0].step(times, ci[0], where='post', color='black', ls='--', lw=1, dashes=(1.5, 0.75))
    ax[0].step(times, ci[1], where='post', color='black', ls='--', lw=1, dashes=(1.5, 0.75))

    times, km, ci, ll_2 = KM(tt_[~lr], ee[~lr], t1=[0, 365, 730, 1095], tmax=1196)
    m2 = np.min(ci[0])
    dd = pd.DataFrame(np.concatenate((times[:, None], km[:, None], ci[0][:, None], ci[1][:, None]), axis=1))
    dd.columns = ['times', 'km', 'lower95', 'upper95']
    dd.to_csv(dir_out + events[cc] + '/data/kmplots2.csv', sep=';')

    ax[0].step(times, km, where='post', color=colormap[cc], lw=1)
    ax[0].step(times, ci[0], where='post', color=colormap[cc], ls='--', lw=1, dashes=(1.5, 0.75))
    ax[0].step(times, ci[1], where='post', color=colormap[cc], ls='--', lw=1, dashes=(1.5, 0.75))

    ax[0].set_xlim([0, 1196])
    ax[0].set_xticks([0, 365, 730, 1095])
    ax[0].set_xticklabels([0, 1, 2, 3])
    ax[0].text(0.05, 0.05, hr_text, size=fontsize, transform=ax[0].transAxes)

    ax[1].text(-0.05, 0.6, str(int(ll_1[0, 1])), color='black', size=fontsize-1)
    ax[1].text(-0.05, 0.2, str(int(ll_2[0, 1])), color='black', weight='bold', size=fontsize-1)
    ax[1].text(-0.05, -0.3, str(int(ll_1[0, 0])), color='black', size=fontsize-1)
    ax[1].text(-0.05, -0.7, str(int(ll_2[0, 0])), color='black', weight='bold', size=fontsize-1)

    ax[1].text(0.24, 0.6, str(int(ll_1[1, 1])), color='black', size=fontsize-1)
    ax[1].text(0.24, 0.2, str(int(ll_2[1, 1])), color='black', weight='bold', size=fontsize-1)
    ax[1].text(0.24, -0.3, str(int(ll_1[1, 0])), color='black', size=fontsize-1)
    ax[1].text(0.24, -0.7, str(int(ll_2[1, 0])), color='black', weight='bold', size=fontsize-1)

    ax[1].text(0.55, 0.6, str(int(ll_1[2, 1])), color='black', size=fontsize-1)
    ax[1].text(0.55, 0.2, str(int(ll_2[2, 1])), color='black', weight='bold', size=fontsize-1)
    ax[1].text(0.55, -0.3, str(int(ll_1[2, 0])), color='black', size=fontsize-1)
    ax[1].text(0.55, -0.7, str(int(ll_2[2, 0])), color='black', weight='bold', size=fontsize-1)

    ax[1].text(0.85, 0.6, str(int(ll_1[3, 1])), color='black', size=fontsize-1)
    ax[1].text(0.85, 0.2, str(int(ll_2[3, 1])), color='black', weight='bold', size=fontsize-1)
    ax[1].text(0.85, -0.3, str(int(ll_1[3, 0])), color='black', size=fontsize-1)
    ax[1].text(0.85, -0.7, str(int(ll_2[3, 0])), color='black', weight='bold', size=fontsize-1)

    ax[1].text(-0.32, 1.4, 'Years:', color='black', size=fontsize-1)
    ax[1].text(-0.32, 0.45, 'At Risk:', color='black', size=fontsize-1)
    ax[1].text(-0.32, -0.45, 'Events:', color='black', size=fontsize-1)
    ax[0].set_ylabel('S(t)')
    ax[1].set_axis_off()

    mm = np.round(np.minimum(m1, m2), 2)
    if mm >= np.minimum(m1, m2):
         mm -= 0.01
    ax[0].set_ylim([mm-0.0005, 1.0005])
    ax[0].set_yticks([1, np.round(1-((1-mm)/2), 2), mm])
    plt.savefig(dir_out + events[cc] + '/figures/km.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/km.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()       

    fig, ax = plt.subplots(1, 1, figsize=(1.1*cm,1.1*cm), dpi=600)
    fig.subplots_adjust(wspace=0.0, hspace=0.4)
        
    times, km, ci, ll_1 = KM(tt_[lr], ee[lr], t1=[0, 365, 730, 1095], tmax=1196)
    m1 = np.min(ci[0])
    ax.step(times, km, where='post', color='black',  lw=0.85)
    ax.fill_between(times, ci[0], ci[1], color='black', alpha=0.15)

    times, km, ci, ll_2 = KM(tt_[~lr], ee[~lr], t1=[0, 365, 730, 1095], tmax=1196)
    m2 = np.min(ci[0])
    ax.step(times, km, where='post', color=colormap[cc], lw=1.15)
    ax.fill_between(times, ci[0], ci[1], color=colormap[cc], alpha=0.15)

    ax.set_xlim([0, 1196])
    ax.set_xticks([0, 1095])
    ax.set_xticklabels(['0', '3'])
    ax.set_title(hr_text_small, fontsize=6)
    
    mm = np.round(np.minimum(m1, m2), 2)
    if mm >= np.minimum(m1, m2):
         mm -= 0.01
    ax.set_ylim([mm-0.0005, 1.0005])
    ax.set_yticks([1, mm])
    
    plt.savefig(dir_out + events[cc] + '/figures/km_small.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/km_small.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    
pd.DataFrame(estll).to_csv(dir_out + '/main/data/kmest.csv')
    

# %%
## Metrics / Calibration
#=======================================================================================================================
concordance=[]
calibration=[]
AUC=[]
Brier = []
predfrac = []
realfrac = []

for cc in tqdm.tqdm(range(22)):
    print(events[cc])
    fpr=[]
    tpr=[]
    roc_auc=[]
    cases_pred=[]
    cases_theo=[]
    
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
    dd.to_csv(dir_out + events[cc] + '/data/metrics_raw.csv', sep=';')
    

    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/users/projects/cancer_risk/main/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/metrics_raw.csv'"

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

    os.remove('/users/projects/cancer_risk/tmp/est_conc.txt')
    
    splits = 10
    step = np.ceil(np.sum(idx)/splits).astype(int)
    idxmat = np.zeros((np.sum(idx), splits))
    for kk in range(splits):
        idxmat[kk*step:(kk+1)*step, kk] = 1
    
    idxsort = np.argsort(pp)
    pp = pp[idxsort]
    y_ = y_[idxsort]
    tt_ = tt_[idxsort]
    
    cases_pred = []
    cases_theo = []
    cases_theo0 = []
    cases_theo1 = []
    for col in range(splits):
        times, km, ci, ll_1 = KM(tt_[idxmat[:, col].astype(bool)], y_[idxmat[:, col].astype(bool)], t1=[0, 365, 730, 1095], tmax=1196)   
        cases_theo.extend([(1-km[-1])*100])
        cases_theo0.extend([(1-ci[0][-1])*100])
        cases_theo1.extend([(1-ci[1][-1])*100])
        cases_pred.extend([pp[idxmat[:, col].astype(bool)].mean()*100])
        
        
    cases_theo = np.asarray(cases_theo)
    cases_pred = np.asarray(cases_pred)
    cases_theo0 = np.asarray(cases_theo0)
    cases_theo1 = np.asarray(cases_theo1)
    predfrac.extend([cases_pred])
    realfrac.extend([cases_theo])
        
    #reg = LinearRegression().fit(cases_pred[:, None], cases_theo)
                       
    fpr, tpr, threshold = metrics.roc_curve(y_, pp)
    roc_auc = metrics.auc(fpr, tpr)
    
    AUC.extend([roc_auc])

    Brier.extend([metrics.brier_score_loss(y_, pp)])
    
    m = np.maximum(np.max([np.max(cases_theo)]), np.max([np.max(cases_pred)]))
                                  
    #txt = 'Intercept: ' + str(np.round(reg.intercept_, 2)) + ', Slope: ' + str(np.round(reg.coef_[0], 2))
    fig, ax = plt.subplots(1, 2, figsize=(8.2*cm, 3.8*cm), dpi=600)
    fig.subplots_adjust(wspace=0.4, hspace=0.5)
        
    ax[1].plot([0, m], [0, m], color='black', ls='--', linewidth=0.5)
    ax[1].scatter(x=cases_pred, y=cases_theo, marker='x', s=8, linewidth=0.7, color=colormap[cc])
    ax[1].plot([cases_pred, cases_pred], [cases_theo0, cases_theo1], linewidth=0.7, color=colormap[cc])
    #ax[1].plot(np.asarray([0, m]), np.asarray([reg.intercept_, reg.intercept_+m*reg.coef_[0]]), color=colormap[cc], linewidth=1)
    ax[1].set_xlabel('Predicted Fraction in %')
    ax[1].set_ylabel('Realized Fraction in %')
    #ax[1].text(0.04, 0.007, txt, size=fontsize-1)
                 
    ax[0].plot(fpr, tpr, color=colormap[cc], lw=1, label='Area: ' + str(np.round(roc_auc, 2)))
    ax[0].plot([0, 1], [0, 1], color='black', ls='--', lw=0.5)          
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate') 
    ax[0].legend(loc=4, prop={'size':fontsize-1}, frameon=False)

    plt.savefig(dir_out + events[cc] + '/figures/metrics.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/metrics.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    pickle.dump({'tpr':tpr,
               'fpr':fpr,
               'cases_pred':cases_pred,
               'cases_theo':cases_theo,
                'cases_theo0':cases_theo0,
               'cases_theo1':cases_theo1,
               'rocauc':roc_auc,
               'concordance':conc}, open(dir_out + events[cc] + '/data/metrics.pickle', 'wb'))
    
    #txt = str(np.round(reg.intercept_, 2)) + ' / ' + str(np.round(reg.coef_[0], 2))
    fig, ax = plt.subplots(1, 1, figsize=(1.1*cm, 1.1*cm), dpi=600)
    ax.plot([0, m], [0, m], color='black', ls='--', linewidth=0.5)
    ax.scatter(x=cases_pred, y=cases_theo, marker='x', s=8, linewidth=0.80, color=colormap[cc])
    #ax.plot(np.asarray([0, m]), np.asarray([reg.intercept_, reg.intercept_+m*reg.coef_[0]]), color=colormap[cc], linewidth=1.15)
    #ax.set_title(txt, fontsize=6)
    
    mm = np.round(np.maximum(cases_pred, cases_theo).max(), 2)
    ax.set_xlim([0, mm*1.07])
    ax.set_ylim([0, mm*1.07])
    ax.set_xticks([0, mm])
    ax.set_yticks([0, mm])

    plt.savefig(dir_out + events[cc] + '/figures/metrics_small.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/metrics_small.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    
dd = pd.DataFrame(concordance)
dd.columns = ['concordance', 'se']
dd.to_csv(dir_out + 'main' + '/data/concordance.csv', sep=';')

dd = pd.DataFrame(np.concatenate([np.zeros((22, 1)), np.asarray(concordance), np.asarray(AUC)[:, None], np.asarray(Brier)[:, None], np.asarray(realfrac), np.asarray(predfrac)], axis=1))
dd.iloc[:, 0] = Events
dd.columns = ['cancer', 'concordance', 'se', 'AUC', 'Brier', 'Obs_Risk_q10', 'Obs_Risk_q20', 'Obs_Risk_q30', 'Obs_Risk_q40', 'Obs_Risk_q50', 'Obs_Risk_q60', 'Obs_Risk_q70', 'Obs_Risk_q80', 'Obs_Risk_q90', 'Obs_Risk_q100','Pred_Risk_q10', 'Pred_Risk_q20', 'Pred_Risk_q30', 'Pred_Risk_q40', 'Pred_Risk_q50', 'Pred_Risk_q60', 'Pred_Risk_q70', 'Pred_Risk_q80', 'Pred_Risk_q90', 'Pred_Risk_q100']
dd.to_csv(dir_out + 'main' + '/tables/metrics.csv', sep=';')


# %%
## Model Tests - Likelihood Ratio Test
#=======================================================================================================================
for cc in tqdm.tqdm(range(22)):
    '''
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
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=pred2[:, :, (0, 2)].sum(axis=2), sex=sex.astype(int)[idx])
    pp_health=[]
    for ii in (range(np.sum(idx))):
        pp_health.extend(cif_(ii))
    pp_health = np.asarray(pp_health)
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=pred2[idx, :, 1], sex=sex.astype(int)[idx])
    pp_gene=[]
    for ii in (range(np.sum(idx))):
        pp_gene.extend(cif_(ii))
    pp_gene = np.asarray(pp_gene)
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=np.zeros_like(pred[idx, 1, :]), sex=sex.astype(int)[idx], full=False)
    pp_base=[]
    for ii in (range(np.sum(idx))):
        pp_base.extend(cif_(ii))
    pp_base = np.asarray(pp_base)
    

    dd = pd.DataFrame(np.concatenate((tt_surv[idx, None], y_[:, None], pp[:, None], pp_health[:, None], pp_gene[:, None], pp_base[:, None], pred2[:, cc, (0, 2)].sum(axis=1)[idx, None], pred2[:, cc, 1][idx, None] ), axis=1).astype(float))
    dd.columns = ['time', 'events', 'all', 'health', 'gene', 'base', 'r_health', 'r_gene']
    dd.to_csv(dir_out + events[cc] + '/data/LR_raw.csv', sep=';')
    '''
    
    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/users/projects/cancer_risk/main/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/LR_raw.csv'"

    c = '''
    dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')


    m1 = coxph(Surv(time, events)~base, data=dd)
    m2 = coxph(Surv(time, events)~base+r_health, data=dd)
    m3 = coxph(Surv(time, events)~base+r_gene, data=dd)
    m4 = coxph(Surv(time, events)~base+r_gene+r_health, data=dd)
    #m5 = coxph(Surv(time, events)~ukbest, data=dd)


    x = paste(m1$loglik[1], m1$loglik[2], m2$loglik[1], m2$loglik[2], m3$loglik[1], m3$loglik[2], m4$loglik[1], m4$loglik[2], sep=";")
    write(x, file=paste('/users/projects/cancer_risk/tmp/est_LR.txt', sep=''), append=FALSE, sep=";")


    x = paste(unname(summary(m1)$concordance[1]), unname(summary(m1)$concordance[2]), unname(summary(m2)$concordance[1]), unname(summary(m2)$concordance[2]), unname(summary(m3)$concordance[1]), unname(summary(m3)$concordance[2]), unname(summary(m4)$concordance[1]), unname(summary(m4)$concordance[2]),sep=';')

    write(x, file=paste('/users/projects/cancer_risk/tmp/est_conc.txt', sep=''), append=FALSE, sep=";")

    # ukbest vs all 
    #p1 = pchisq(-2 * (m5$loglik[2] - m4$loglik[2]), 2, lower.tail=FALSE)

    # base vs health
    p2 = pchisq(-2 * (m1$loglik[2] - m2$loglik[2]), 1, lower.tail=FALSE)

    # base vs gene 
    p3 = pchisq(-2 *(m1$loglik[2] - m3$loglik[2]), 1, lower.tail=FALSE)

    # all vs gene 
    p4 = pchisq(-2 * (m2$loglik[2] - m4$loglik[2]), 1, lower.tail=FALSE)

    # all vs heath 
    p5 = pchisq(-2 * (m3$loglik[2] - m4$loglik[2]), 1, lower.tail=FALSE)

    # all vs base 
    p6 = pchisq(-2 * (m1$loglik[2] - m4$loglik[2]), 2, lower.tail=FALSE)

    p <- c(p2, p3, p4, p5, p6)
    #res <- c()
    #for(padj in p.adjust.methods){res <- c(res, p.adjust(p, method=padj, n = length(p)))}
    dd = as.data.frame(matrix(p, ncol=5, byrow=TRUE))
    #row.names(dd) <- p.adjust.methods
    names(dd) <- c("base_health", "base_gene", "all_health", "all_gene", "all_base")
    write.csv(dd, "/users/projects/cancer_risk/tmp/est_pvalues.csv")

    '''

    with open('/users/projects/cancer_risk/tmp/est_LR.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/users/projects/cancer_risk/tmp/est_LR.R'], shell=False)
    os.remove('/users/projects/cancer_risk/tmp/est_LR.R')

    LR_est = np.squeeze(np.loadtxt('/users/projects/cancer_risk/tmp/est_LR.txt', delimiter=';'))
    dd = pd.DataFrame(LR_est[None, :])

    dd.columns = ['logL_01', 'logL_base','logL_02', 'logL_base_health', 'logL_03', 'logL_base_gene', 'logL_04', 'logL_all']
    dd.to_csv(dir_out + events[cc] + '/data/LR.csv', sep=';')

    conc_est = np.squeeze(np.loadtxt('/users/projects/cancer_risk/tmp/est_conc.txt', delimiter=';'))
    dd = pd.DataFrame(conc_est[None, :])
    dd.columns = ['concordance_base', 'se_base','concordance_base_health', 'se_base_health', 'concordance_base_gene', 'se_base_gene', 'concordance_all', 'se_all']
    dd.to_csv(dir_out + events[cc] + '/data/concordance_sub.csv', sep=';')

    pval_est = pd.read_csv('/users/projects/cancer_risk/tmp/est_pvalues.csv')
    pval_est.to_csv(dir_out + events[cc] + '/data/pvalues.csv', sep=';')

    os.remove('/users/projects/cancer_risk/tmp/est_LR.txt')
        
## PostProcessing
#=====================================================================================================================
dd1 = pd.read_csv(dir_out + events[0] + '/data/LR.csv', sep=';')
dd2 = pd.read_csv(dir_out + events[0] + '/data/pvalues.csv', sep=';')
dd3 = pd.read_csv(dir_out + events[0] + '/data/concordance_sub.csv', sep=';')

dd = pd.DataFrame(np.concatenate((

np.asarray(dd1.loc[:, ['Unnamed: 0', 'logL_base', 'logL_base_health', 'logL_base_gene', 'logL_all']]),
np.asarray(dd2.iloc[:, 2:]),
np.asarray(dd3.iloc[:, 1:])
), axis=1))

for cc in range(1, 22):
    dd1 = pd.read_csv(dir_out + events[cc] + '/data/LR.csv', sep=';')
    dd2 = pd.read_csv(dir_out + events[cc] + '/data/pvalues.csv', sep=';')
    dd3 = pd.read_csv(dir_out + events[cc] + '/data/concordance_sub.csv', sep=';')
    
    dd_helpvar = pd.DataFrame(np.concatenate((

    np.asarray(dd1.loc[:, ['Unnamed: 0', 'logL_base', 'logL_base_health', 'logL_base_gene', 'logL_all']]),
    np.asarray(dd2.iloc[:, 2:]),
    np.asarray(dd3.iloc[:, 1:])
    ), axis=1))

    dd = dd.append(dd_helpvar)

dd.columns = ['cancer', 'logL_base', 'logL_base_health', 'logL_base_gene', 'logL_all', 'pvalue_base_health', 'pvalue_base_gene', 'pvalue_all_health', 'pvalue_all_gene', 'pvalue_all_base', 'concordance_base', 'se_base','concordance_base_health', 'se_base_health', 'concordance_base_gene', 'se_base_gene', 'concordance_all', 'se_all']

dd.iloc[:, 0] = Events

dd.to_csv(dir_out + 'main' + '/tables/model_comparison.csv')

# %%
## Predictor Distribution
#=======================================================================================================================
for cc in tqdm.tqdm(range(22)):
    pred[:, 1, cc] = pred2[:, cc, 0]  + pred2[:, cc, 2] +  pred2[:, cc, 1]
    fig, ax = plt.subplots(1, 2, figsize=(7*cm, 3*cm), dpi=600, sharey=True, gridspec_kw={'width_ratios':[3, 1]})
    fig.subplots_adjust(wspace=0.05)
    ax[1].hist([pred[pred[:, 0, cc]==0, 1, cc], pred[pred[:, 0, cc]==1, 1, cc]], color=['.1', colormap[cc]], density=True, label=['Healthy', 'Cancer'], orientation="horizontal")
    ax[1].legend(frameon=False, fontsize=5)
    ax[1].set_xlabel('Freq.')
    idxcc = np.sort(np.random.choice(np.where(pred[:, 0, cc]==0)[0], 50000, replace=False))
    ax[0].plot((age/365)[idxcc], pred[idxcc, 1, cc], ls='', marker='x', markersize=0.75, markeredgewidth=0.05, color='.1')
    try:
        idxcc = np.random.choice(np.where(pred[:, 0, cc]==1)[0], 2500)
    except:
        idxcc = pred[:, 0, cc]==1
    ax[0].plot((age/365)[idxcc], pred[idxcc, 1, cc], ls='', marker='x', markersize=1, markeredgewidth=0.15, color=colormap[cc])
    ax[0].set_ylabel('Log(Hazard)')
    ax[0].set_xlabel('Age')
    plt.savefig(dir_out + events[cc] + '/figures/predictor_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/predictor_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    

# %%
## Concordance Plot
#=======================================================================================================================
conc = pd.read_csv(dir_out + 'main' + '/data/concordance.csv', sep=';')
conc2 = pd.read_csv(dir_out + 'main' + '/data/concordance2.csv', sep=';')
concordance = np.asarray(conc.iloc[:, 1:])
concordance2 = np.asarray(conc2.iloc[:, -2:])
ll = [round_(concordance[ii, 0], 2) + ' (' + round_(concordance[ii, 1], 2) + ')' + '\n' + round_(concordance2[ii, 0], 2) + ' (' + round_(concordance2[ii, 1], 2) + ')' for ii in range(22)]

fig, ax = plt.subplots(1, 1, figsize=(3.3*cm, 15*cm), dpi=600)
ax.barh(width=concordance[:, 0], y=np.arange(22)+0.22, color=colormap, edgecolor=colormap ,lw=0.5, height=0.4, xerr=concordance[:, 1])
ax.barh(width=concordance2[:, 0], y=np.arange(22)-0.22, color='white', edgecolor=colormap ,lw=0.5, hatch='xxxxxxxxxxxx', height=0.4, xerr=concordance2[:, 1])

ax.set_yticks(np.arange(22))
ax.set_yticklabels(Events)
ax.set_xlim(0.5, 1)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xticklabels(['0.5', '', '0.7', '', '0.9', ''])
ax.set_ylim([-0.8, 21.8])
ax0 = ax.twinx()
ax0.set_ylim(ax.get_ylim())
ax0.set_yticks(np.arange(22))

ax0.set_yticklabels(ll)
ax.set_xlabel('Concordance')

plt.savefig(dir_out + 'main' + '/figures/concordance_summary.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out + 'main' + '/figures/concordance_summary.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# %%
print('finished')
exit()



