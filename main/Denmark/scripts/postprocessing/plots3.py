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
from scipy.stats.mstats import gmean

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


## Data
#=======================================================================================================================

disease_codes = pd.read_csv(dir_data + 'CancerRisk/disease_ref.csv', sep=';')
disease_codes = np.asarray(disease_codes.iloc[:, 1:])
disease_cat = np.load(dir_data + 'CancerRisk/disease_codes_cat.npy')

#run_id = int(sys.argv[1])

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
#Apred = []
'''
for ii in tqdm.tqdm(range(1000)):
    ii1 = ii//100
    ii2 = (ii - ii1*100)//10
    file = dir_data + 'predictions_ukb/f_%i/f_%i/s_%i' %(ii1, ii2, ii)
    with h5py.File(file, 'r') as f:
        idx_list = list(f.keys())
        for eid in idx_list:
            time.extend(f[str(eid)]['time'][:, :].tolist())
            pred.extend(f[str(eid)]['pred'][:, :].tolist())
            #Apred.extend(f[str(eid)]['dyn_absolute_risk'][:, :, :].tolist())
'''
file = dir_data + 'predictions_ukb/master_quick.h5'
with h5py.File(file, 'r') as f:
    time = f['time'][:]
    pred = f['pred'][:]
            
time = np.asarray(time).astype(float)
pred = np.asarray(pred).astype(float)

sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)
out = pred[:, 0, :].copy()
N = time.shape[0]


# %%
## Relative Frequency Plot - matched
#=======================================================================================================================
file = dir_data + '/frequency/master.h5'
with h5py.File(file, 'r') as ff:
    cc = run_id
    if cc in [7, 8, 9, 10]:
        idx = np.logical_and(~sex, out[:, cc]==1)
        idx2 = np.unique(np.concatenate([np.random.choice(np.where(np.logical_and(~idx, np.logical_and(np.logical_and(age >= age[ii] - 365, age < age[ii] + 365), sex==sex[ii])))[0], 100) for ii in np.where(idx)[0]]))

        dfre = np.zeros((1305,)) + 10e-100
        for ii in np.where(idx)[0]:
            dfre += ff['data'][ii, :]
        dfre = dfre/np.where(idx)[0].shape[0]

        hfre = np.zeros((1305,)) + 10e-100
        for ii in idx2:
            hfre += ff['data'][ii, :]
        hfre = hfre/idx2.shape[0]

        #dfre = ff['data'][idx, :].mean(axis=0) + 10e-100
        #idx_dfre = dfre>=0.01
        #hfre = ff['data'][idx2, :].mean(axis=0) + 10e-100

        rr = np.round(dfre/hfre, 5)
        #rr[~idx_dfre] = 1
        #idx = np.logical_or(rr>1.01, rr<0.99)
        dd = pd.DataFrame(np.concatenate((disease_codes[:], rr[:, None], dfre[:, None], hfre[:, None], np.repeat(np.where(idx)[0].shape[0], 1305)[:, None], np.repeat(idx2.shape[0], 1305)[:, None]), axis=1))
        dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq', 'Cancer_count', 'Healthy_count']
        #dd.sort_values('RR', inplace=True)
        dd.to_csv(dir_out + events[cc] + '/tables/relative_risk_matched.csv')

    elif cc in [11, 12]:
        idx = np.logical_and(sex,out[:, cc]==1)
        idx2 = np.unique(np.concatenate([np.random.choice(np.where(np.logical_and(~idx, np.logical_and(np.logical_and(age >= age[ii] - 365, age < age[ii] + 365), sex==sex[ii])))[0], 100) for ii in np.where(idx)[0]]))

        dfre = ff['data'][idx, :].mean(axis=0) + 10e-100
        #idx_dfre = dfre>=0.01
        hfre = ff['data'][idx2, :].mean(axis=0) + 10e-100

        rr = np.round(dfre/hfre, 5)
        #rr[~idx_dfre] = 1
        #idx = np.logical_or(rr>1.01, rr<0.99)
        dd = pd.DataFrame(np.concatenate((disease_codes[:], rr[:, None], dfre[:, None], hfre[:, None], np.repeat(np.where(idx)[0].shape[0], 1305)[:, None], np.repeat(idx2.shape[0], 1305)[:, None]), axis=1))
        dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq', 'Cancer_count', 'Healthy_count']
        #dd.sort_values('RR', inplace=True)
        dd.to_csv(dir_out + events[cc] + '/tables/relative_risk_matched.csv')

    else:
        idx = out[:, cc]==1
        idx2 = np.unique(np.concatenate([np.random.choice(np.where(np.logical_and(~idx, np.logical_and(np.logical_and(age >= age[ii] - 365, age < age[ii] + 365), sex==sex[ii])))[0], 100) for ii in np.where(idx)[0]]))

        dfre = ff['data'][idx, :].mean(axis=0) + 10e-100
        #idx_dfre = dfre>=0.01
        hfre = ff['data'][idx2, :].mean(axis=0) + 10e-100

        rr = np.round(dfre/hfre, 5)
        #rr[~idx_dfre] = 1
        #idx = np.logical_or(rr>1.01, rr<0.99)
        dd = pd.DataFrame(np.concatenate((disease_codes[:], rr[:, None], dfre[:, None], hfre[:, None], np.repeat(np.where(idx)[0].shape[0], 1305)[:, None], np.repeat(idx2.shape[0], 1305)[:, None]), axis=1))
        dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq', 'Cancer_count', 'Healthy_count']
        #dd.sort_values('RR', inplace=True)
        dd.to_csv(dir_out + events[cc] + '/tables/relative_risk_matched.csv')



# %%
print('finished')
exit()

# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N plots3
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=20gb
#PBS -l walltime=150:00:00

cd $PBS_O_WORDIR
module load anaconda3/2019.10
source conda activate
module load tools
module load gcc/10.2.0
module load intel/perflibs/2018
module load R/4.1.0

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/postprocessing/plots3.ipynb --output /users/projects/cancer_risk/main/scripts/postprocessing/plots3

/services/tools/anaconda3/2019.10/bin/python3.7 /users/projects/cancer_risk/main/scripts/postprocessing/plots3.py $VAR1
' >> run.sh
for ii in 21; do qsub -v VAR1=$ii run.sh; done


# %%
# Specific for Forest plot - 10% significance level
for cc in tqdm.tqdm(range(20)):
    dd = np.asarray(pd.read_csv(dir_out + events[cc] + '/tables/relative_risk_matched.csv', usecols=[5, 6]))
    dfre = dd[:, 0]
    hfre = dd[:, 1]
    rr = np.round(dfre/hfre, 6)

    mm = tt[cc]['model']
    gg = tt[cc]['guide']
    theta_dnpr_lower, theta_dnpr, theta_dnpr_upper  = gg.quantiles([0.05, 0.5, 0.95])['theta_dnpr']
    theta_dnpr_lower = theta_dnpr_lower.detach().numpy()
    theta_dnpr = theta_dnpr.detach().numpy()
    theta_dnpr_upper = theta_dnpr_upper.detach().numpy()
    sig = np.sign(theta_dnpr_lower) == np.sign(theta_dnpr_upper) 

    theta_sig = theta_dnpr[sig]

    rr_sig = rr[sig[0, :]]
    disease_codes_sig = disease_codes[sig[0, :]]
    dfre_sig = dfre[sig[0, :]]
    hfre_sig = hfre[sig[0, :]]

    rr_sig = rr_sig[np.argsort(-theta_sig)]
    disease_codes_sig = disease_codes_sig[np.argsort(-theta_sig)]
    dfre_sig = dfre_sig[np.argsort(-theta_sig)]
    hfre_sig = hfre_sig[np.argsort(-theta_sig)]

    dd = pd.DataFrame(np.concatenate((disease_codes_sig, rr_sig[:, None], dfre_sig[:, None], hfre_sig[:, None]), axis=1))
    dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq']
    dd.to_csv(dir_out + events[cc] + '/tables/forest_relative_risk_matched.csv')
    dd = pd.DataFrame(np.concatenate((disease_codes_sig[:, 0, None], rr_sig[:, None], dfre_sig[:, None], hfre_sig[:, None]), axis=1))
    dd.columns = ['icd10', 'RR', 'Cancer_Freq', 'Healthy_Freq']
    dd.to_csv(dir_out + events[cc] + '/tables/forest_relative_risk_matched_small.csv')


# %%
## Relative Frequency Plot - 1
#=======================================================================================================================

# Main Figure
theta_dnpr_lower1 = []
theta_dnpr_lower5 = []
theta_dnpr_lower10 = []
theta_dnpr = []
theta_dnpr_upper10 = []
theta_dnpr_upper5 = []
theta_dnpr_upper1 = []
sig_dnpr1 = []
sig_dnpr5 = []
sig_dnpr10 = []

theta_gene_lower1 = []
theta_gene_lower5 = []
theta_gene_lower10 = []
theta_gene = []
theta_gene_upper10 = []
theta_gene_upper5 = []
theta_gene_upper1 = []
sig_gene1 = []
sig_gene5 = []
sig_gene10 = []

theta_bth_lower1 = []
theta_bth_lower5 = []
theta_bth_lower10 = []
theta_bth = []
theta_bth_upper10 = []
theta_bth_upper5 = []
theta_bth_upper1 = []
sig_bth1 = []
sig_bth5 = []
sig_bth10 = []

for cc in tqdm.tqdm(range(22)):
    mm = tt[cc]['model']
    gg = tt[cc]['guide']

    a, b, c, d, e, f, g  = gg.quantiles([0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])['theta_dnpr']
    a = a.detach().numpy()
    b = b.detach().numpy()
    c = c.detach().numpy()
    d = d.detach().numpy()
    e = e.detach().numpy()
    f = f.detach().numpy()
    g = g.detach().numpy()

    h = np.sign(a) == np.sign(g) 
    i = np.sign(b) == np.sign(f) 
    j = np.sign(c) == np.sign(e) 

    theta_dnpr_lower1.extend(a)
    theta_dnpr_lower5.extend(b)
    theta_dnpr_lower10.extend(c)
    theta_dnpr.extend(d)
    theta_dnpr_upper10.extend(e)
    theta_dnpr_upper5.extend(f)
    theta_dnpr_upper1.extend(g)
    sig_dnpr1.extend(h)
    sig_dnpr5.extend(i)
    sig_dnpr10.extend(j)

    a, b, c, d, e, f, g  = gg.quantiles([0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])['theta_gene']
    a = a.detach().numpy()
    b = b.detach().numpy()
    c = c.detach().numpy()
    d = d.detach().numpy()
    e = e.detach().numpy()
    f = f.detach().numpy()
    g = g.detach().numpy()

    h = np.sign(a) == np.sign(g) 
    i = np.sign(b) == np.sign(f) 
    j = np.sign(c) == np.sign(e) 

    theta_gene_lower1.extend(a)
    theta_gene_lower5.extend(b)
    theta_gene_lower10.extend(c)
    theta_gene.extend(d)
    theta_gene_upper10.extend(e)
    theta_gene_upper5.extend(f)
    theta_gene_upper1.extend(g)
    sig_gene1.extend(h)
    sig_gene5.extend(i)
    sig_gene10.extend(j)


    a, b, c, d, e, f, g  = gg.quantiles([0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])['theta_bth']
    a = a.detach().numpy()
    b = b.detach().numpy()
    c = c.detach().numpy()
    d = d.detach().numpy()
    e = e.detach().numpy()
    f = f.detach().numpy()
    g = g.detach().numpy()

    h = np.sign(a) == np.sign(g) 
    i = np.sign(b) == np.sign(f) 
    j = np.sign(c) == np.sign(e) 

    theta_bth_lower1.extend(a)
    theta_bth_lower5.extend(b)
    theta_bth_lower10.extend(c)
    theta_bth.extend(d)
    theta_bth_upper10.extend(e)
    theta_bth_upper5.extend(f)
    theta_bth_upper1.extend(g)
    sig_bth1.extend(h)
    sig_bth5.extend(i)
    sig_bth10.extend(j)

theta_dnpr_lower1 = np.stack((theta_dnpr_lower1), axis=1)
theta_dnpr_lower5 = np.stack((theta_dnpr_lower5), axis=1)
theta_dnpr_lower10 = np.stack((theta_dnpr_lower10), axis=1)
theta_dnpr = np.stack((theta_dnpr), axis=1)
theta_dnpr_upper10 = np.stack((theta_dnpr_upper10), axis=1)
theta_dnpr_upper5 = np.stack((theta_dnpr_upper5), axis=1)
theta_dnpr_upper1 = np.stack((theta_dnpr_upper1), axis=1)
sig_dnpr1 = np.stack((sig_dnpr1), axis=1)
sig_dnpr5 = np.stack((sig_dnpr5), axis=1)
sig_dnpr10 = np.stack((sig_dnpr10), axis=1)

theta_gene_lower1 = np.stack((theta_gene_lower1), axis=1)
theta_gene_lower5 = np.stack((theta_gene_lower5), axis=1)
theta_gene_lower10 = np.stack((theta_gene_lower10), axis=1)
theta_gene = np.stack((theta_gene), axis=1)
theta_gene_upper10 = np.stack((theta_gene_upper10), axis=1)
theta_gene_upper5 = np.stack((theta_gene_upper5), axis=1)
theta_gene_upper1 = np.stack((theta_gene_upper1), axis=1)
sig_gene1 = np.stack((sig_gene1), axis=1)
sig_gene5 = np.stack((sig_gene5), axis=1)
sig_gene10 = np.stack((sig_gene10), axis=1)

theta_bth_lower1 = np.stack((theta_bth_lower1), axis=1)
theta_bth_lower5 = np.stack((theta_bth_lower5), axis=1)
theta_bth_lower10 = np.stack((theta_bth_lower10), axis=1)
theta_bth = np.stack((theta_bth), axis=1)
theta_bth_upper10 = np.stack((theta_bth_upper10), axis=1)
theta_bth_upper5 = np.stack((theta_bth_upper5), axis=1)
theta_bth_upper1 = np.stack((theta_bth_upper1), axis=1)
sig_bth1 = np.stack((sig_bth1), axis=1)
sig_bth5 = np.stack((sig_bth5), axis=1)
sig_bth10 = np.stack((sig_bth10), axis=1)


#bthnames = np.asarray(['Alcohol', 'Smoking', 'High BP.', 'Low BP.', 'Height', 'Weight', 'Age at first Birth']).astype(object)
#genenames = np.concatenate([np.repeat(Events[ii], 4).tolist() for ii in range(20)]).astype(object)

idx_dnpr = (sig_dnpr10.sum(axis=1) >3)
#idx_gene = (sig_gene10.sum(axis=1) >3) irrelevant
#idx_bth = (sig_bth10.sum(axis=1) >3)

effect = np.exp(theta_dnpr[idx_dnpr])
sig = sig_dnpr10[idx_dnpr, :]
names = disease_codes[idx_dnpr, 0] + ' ' + disease_codes[idx_dnpr, 1]

center_ = gmean(effect, axis=1)
idxsort = np.argsort(center_)

effect = effect[idxsort]
names = names[idxsort]
center_ = center_[idxsort]
sig = sig[idxsort, :]
min_ = effect.min(axis=1)
max_ = effect.max(axis=1)

# plot 1
dd = pd.DataFrame(np.concatenate((names[:, None],
            effect), axis=1))
nn = ['names']
nn.extend(events)
dd.columns = nn
dd.to_csv(dir_out + 'main' + '/data/pancan_disease_selection.csv')

# adjust names - for space reasons
#names[8] = 'I73 Peripheral vascular diseases'
#names[9] = 'E11 Non-insulin diabetes mellitus'
#names[10] = 'L02 Cutaneous abscess, furuncle, carbuncle'
#names[-2] = 'F10 Mental disorders due to alcohol'
#names[-5] = 'J44 Other COPD'

fig, ax = plt.subplots(1, 1, figsize=(3*cm, 7*cm), dpi=600)
for cc in range(20):
    ax.plot(effect[:, cc], range(effect.shape[0]), ls='', marker='.', markersize=3 , color=colormap[cc])
    

    for jj in range(effect.shape[0]):
        if sig[jj, cc]:
            ax.plot(effect[jj, cc], jj, ls='', marker='x', markersize=3, color=colormap[cc])
    
        
    ax.plot(center_, range(effect.shape[0]), ls='', marker='|', markersize=6, color='black')


ax.set_yticks(np.arange(effect.shape[0]))
ax.set_yticklabels(names)
ax0 = ax.twinx()
ax0.set_ylim(ax.get_ylim())
ax0.set_yticks(np.arange(effect.shape[0]))
ax0.set_yticklabels(['('+ str(round_(min_[ii], 2)) + ' < ' + str(round_(center_[ii], 2)) + ' > ' + str(round_(max_[ii], 2)) + ')'
 for ii in range(effect.shape[0])])
ax.set_xlabel('Hazard')
ax.axvline(1, ls=(0, (1, 1)), color='black', lw=0.75)

plt.savefig(dir_out + 'main' + '/figures/forest.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out + 'main' + '/figures/forest.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()


# %%
## Relative Frequency Plot - 2
#=======================================================================================================================

file = dir_data + 'predictions_ukb/master_quick.h5'
with h5py.File(file, 'r') as f:
    out = f['pred'][:, 0, :]

dfre = []
hfre = []
for cc in tqdm.tqdm(range(20)):
    dd = np.asarray(pd.read_csv(dir_out + events[cc] + '/tables/relative_risk_matched.csv', usecols=[5, 6]))
    dfre.extend([dd[:, 0].tolist()])
    hfre.extend([dd[:, 1].tolist()])

dfre = np.asarray(dfre).T       
hfre = np.asarray(hfre).T
event_count = (out[:, :20]==1).sum(axis=0)[None, :]    

dfre2 = dfre[idx_dnpr].copy()
dfre2 = np.concatenate((dfre2, np.zeros((3, 20))))
dfre2 = dfre2[idxsort]

hfre2 = hfre[idx_dnpr].copy()
hfre2 = np.concatenate((hfre2, np.zeros((3, 20))))
hfre2 = hfre2[idxsort]

names = disease_codes[idx_dnpr, 1].copy()
names = np.concatenate((names, np.repeat('', 3)))
names = names[idxsort]

pd.DataFrame(np.concatenate((names[:, None], dfre2, hfre2), axis=1)).to_csv(dir_out + 'main' + '/data/dfreq_matched.csv')
pd.DataFrame(np.concatenate((names[:, None], dfre2*event_count, hfre2*event_count), axis=1)).to_csv(dir_out + 'main' + '/data/dfreq_count_matched.csv')

att_cancer = (dfre2*event_count)
bottom = np.zeros((18,))
fig, ax = plt.subplots(1, 1, figsize=(3*cm,8*cm), dpi=300)
for cc in range(20):
    ax.barh(range(18), width=att_cancer[:, cc], left=bottom, color=colormap[cc])
    bottom += att_cancer[:, cc]

ax.plot((hfre2*event_count)[:, :20].sum(axis=1), range(18), ls='', marker='|', markersize=9, color='black', label='Expected Cases at Healthy rate') 
#ax.set_yticks(np.arange(effect.shape[0]))
ax.set_ylim([-0.8, 17.8])
ax.set_yticklabels([])
ax.set_yticks(range(18))
ax.set_xlabel('Attributable Cases')
#ax.legend(frameon=False, fontsize=5)
plt.savefig(dir_out + 'main' + '/figures/disease_freq_matched.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out + 'main' + '/figures/disease_freq_matched.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   



