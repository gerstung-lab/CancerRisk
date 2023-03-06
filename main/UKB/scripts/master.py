# %%
## Custom Modules
#=======================================================================================================================
import sys
import os 
import tqdm
import h5py
import dill
import subprocess
import time
import pickle
from _custom_functions import KM, CIF, metric_table, round_
from _plots import Age_Sex_plot, risk_plot_5yr, cumhaz_plot 

import numpy as np 
import pandas as pd 
from sklearn import metrics
from scipy import interp
from sklearn.linear_model import LinearRegression
from scipy.stats.mstats import gmean

from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

sys.path.append(ROOT_DIR + 'projects/CancerRisk/scripts/model')
from m1 import predictor

events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']


Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Melanoma', 'Breast', 
                'Cervix Uteri', 'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney', 'Bladder', 'Brain',
                'Thyroid', 'NHL', 'MM', 'AML', 'Other', 'Death']

for cc in range(22):
    print(cc, events[cc])

disease_codes = np.load(ROOT_DIR + 'projects/CancerRisk/data/prep/disease_codes.npy', allow_pickle=True)

# %%
disease_codes.shape

# %%
## Model
#=======================================================================================================================
tt = [pickle.load(open(ROOT_DIR + 'projects/CancerRisk/model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]
A0 = np.load(ROOT_DIR + 'projects/CancerRisk/model/' + 'all' + '/breslow.npy')
A0_base = np.load(ROOT_DIR + 'projects/CancerRisk/model/' + 'all' + '/ukb_agesex.npy')     

# %%
tt

# %%


# %%


# %%
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
## Extract Data
#======================================================================================================================
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


pred2 = []
for run_id in tqdm.tqdm(range(101)):
    with h5py.File(ROOT_DIR + 'projects/CancerRisk/data/main/predictions/ukb2_' + str(run_id) + '.h5', 'r') as f:
        ll = list(f.keys())
        for ii in ll:
            pred2.extend([f[ii]['pred_sub'][:, :].tolist()]) 
pred2 = np.asarray(pred2).astype(float)


'''
dcodes = []
for run_id in tqdm.tqdm(range(101)):
    with h5py.File(ROOT_DIR + 'projects/CancerRisk/data/main/disease_freq/ukb_' + str(run_id) + '.h5', 'r') as f:
        ll = list(f.keys())
        for ii in ll:
            dcodes.extend(f[ii]['X'][:, :].tolist())
dcodes = np.asarray(dcodes).astype(float)
dcodes = np.minimum(dcodes, 1) 
'''

'''
Apred = []
for run_id in tqdm.tqdm(range(252)):
    with h5py.File(ROOT_DIR + 'projects/CancerRisk/data/main/predictions_dynamic/ukb_' + str(run_id) + '.h5', 'r') as f:
        ll = list(f.keys())
        for ii in ll:
            Apred.extend(f[ii]['absolute_risk'][:, :, :].tolist())
'''

tt_surv = time[:, 0]
sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)
out = pred[:, 0, :].copy()
predage = np.arange(18250, 25581, 31)


# collect all predictions - 5yrs
prediction=[]
for cc in (range(22)):
    print(events[cc])
    idx = np.logical_or(sex, ~sex)
    ee = pred[idx, 0, cc].copy()
    y_ = ee
    tt_ = time[:, 0][idx].copy()
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1825, A0=A0, pred=pred[idx, 1, :], sex=sex.astype(int)[idx])
    pp=[]
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    prediction.extend([pp[:, None]])    
prediction = np.concatenate((prediction), axis=1)
# adjust sex specific cancers
prediction[~sex, 11:13] = 0
prediction[sex, 7:11] = 0    


# %%
## Basic Plots
#======================================================================================================================
for cc in range(22):
    Age_Sex_plot(pred=pred, sex=sex, age=age, cc=cc)
    risk_plot_5yr(A0=A0, cc=cc)
    cumhaz_plot(A0=A0, cc=cc)

# %%
## Attribution
#======================================================================================================================
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

sig_dnpr_cut = np.abs(theta_dnpr) >= np.log(1.1)
sig_gene_cut = np.abs(theta_gene) >= np.log(1.05) # adjusted for different scale 
sig_bth_cut = np.abs(theta_bth) >= np.log(1.1)

sig_dnpr_cut = np.logical_or(sig_dnpr_cut, sig_dnpr10)
sig_gene_cut = np.logical_or(sig_gene_cut, sig_gene10)
sig_bth_cut = np.logical_or(sig_bth_cut, sig_bth10)


# plot 1
dd = pd.DataFrame(np.concatenate((np.zeros((22, 1)),
            sig_dnpr_cut.sum(axis=0)[:, None],
            sig_gene_cut.sum(axis=0)[:, None],
            sig_bth_cut.sum(axis=0)[:, None], 
            sig_dnpr10.sum(axis=0)[:, None],
            sig_gene10.sum(axis=0)[:, None],
            sig_bth10.sum(axis=0)[:, None]), axis=1))
dd.iloc[:, 0] = Events
dd.columns = ['cancer', 'dnpr_cut', 'gene_cut', 'bth_cut', 'dnpr10', 'gene10', 'bth10']
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_selected_variables.csv')

# Explained contribution by 10% posterior of all
dd = pd.DataFrame(
    np.concatenate((np.zeros((3, 1)), 
    np.concatenate((
np.abs(theta_dnpr * sig_dnpr10).sum(axis=0)/np.abs(theta_dnpr).sum(axis=0)[None, :], 
np.abs(theta_gene * sig_gene10).sum(axis=0)/np.abs(theta_gene).sum(axis=0)[None, :], 
np.abs(theta_bth * sig_bth10).sum(axis=0)/np.abs(theta_bth).sum(axis=0)[None, :]))), axis=1))
dd.iloc[:, 0] = ['DNPR', 'Genealogy', 'BTH']
names = ['']
names.extend(events)
dd.columns = names
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/tables/attribution_explained_contribution10.csv')

idx_sort = np.argsort(-sig_dnpr_cut.sum(axis=0))
fig, ax = plt.subplots(3, 1, figsize=(7.5*cm,12*cm), dpi=600, sharex=True, gridspec_kw={'height_ratios': [3, 2, 3]} )
fig.subplots_adjust(hspace=0.35)
ax[0].bar(np.arange(22), np.minimum(100, sig_dnpr_cut.sum(axis=0))[idx_sort], color=colormap[idx_sort])
ax[0].plot(np.arange(22), sig_dnpr10.sum(axis=0)[idx_sort], color='black', ls='', marker='_', markersize=6)

ax[1].bar(np.arange(22), sig_bth_cut.sum(axis=0)[idx_sort], lw=0.05, color='white', edgecolor=colormap[idx_sort], hatch='xxxxxxxxxx')
ax[1].plot(np.arange(22), sig_bth10.sum(axis=0)[idx_sort], color='black', ls='', marker='_', markersize=6)

ax[2].bar(np.arange(22), sig_gene_cut.sum(axis=0)[idx_sort], color=colormap[idx_sort], lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx')
ax[2].plot(np.arange(22), sig_gene10.sum(axis=0)[idx_sort], color='white', ls='', marker='_', markersize=6)

#ax[0].set_ylabel('Disease Codes')
#ax[0].set_yticks([0, 15, 30])
#ax[0].set_yticklabels([0, 15, 30])


#ax[2].set_ylabel('BTH')
ax[1].set_yticks([0, 2, 4])
ax[1].set_yticklabels([0, 2, 4])


#ax[1].set_ylabel('Genealogy')
ax[2].set_yticks([0, 10, 20])
ax[2].set_yticklabels([0, 10, 20])


ax[2].set_xticks(np.arange(22))
ax[2].set_xticklabels(np.asarray(Events)[idx_sort], rotation=90)


plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution1.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution1.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()              

# plot 2
helpvar1 = sig_dnpr_cut * disease_codes[:, 2, None]
helpvar2 = sig_dnpr10 * disease_codes[:, 2, None]

a, b = np.unique(helpvar1, return_counts=True)
c, d = np.unique(helpvar2, return_counts=True)
a = a[1:]
b = b[1:]
c = c[1:]
d = d[1:]
# shorten some chapter names

a_ = np.asarray(['Perinatal period', 'Infectious diseases', 'Congenital malformations', 'Blood', 'Circulatory system', 'Digestive system', 'Ear and mastoid process', 'Eye and adnexa', 'Genitourinary system', 'Musculoskeletal system', 'Nervous system', 'Respiratory system', 'Skin & tissue', 'Endocrine diseases', 'Mental disorders', 'Neoplasms', 'Pregnancy & childbirth', 'Laboratory findings'])

dd = pd.DataFrame(np.concatenate((
            a[:, None],
            b[:, None],
            c[:, None],
            d[:, None]), axis=1))
dd.columns = ['chapter_cut', 'count_cut', 'chapter_10', 'count_10']
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_disease_chapters.csv')

idx_sort = np.argsort(-b)
fig, ax = plt.subplots(1, 1, figsize=(1.5*cm,4.5*cm), dpi=600, sharex=True)
fig.subplots_adjust(hspace=0.05)
ax.barh(np.arange(18), b[idx_sort], color='.9')

ll = np.zeros((18, ))

for cc in range(22):
    helpvar1 = sig_dnpr_cut * disease_codes[:, 2, None]
    helpvar1 = helpvar1[:, cc] 
    a, b = np.unique(helpvar1, return_counts=True)
    dd_ = pd.DataFrame(np.concatenate((
                a[:, None],
                b[:, None]), axis=1))
    dd_.columns = ['chapter_cut', 'count_cut_' + events[cc]]
    dd_ = dd_.iloc[1:, :]
    dd = dd.merge(dd_, how='left', on='chapter_cut')
    dd.fillna(0, inplace=True)
    b = np.asarray(dd.iloc[:, -1])

    ax.barh(np.arange(18), b[idx_sort], left=ll[idx_sort], color=colormap[cc])
    ll += b

ax.plot(d[idx_sort], np.arange(18), color='black', ls='', marker='|', markersize=6)

ax.yaxis.tick_right()
ax.set_yticks(np.arange(18))
ax.set_yticklabels(np.asarray(a_)[idx_sort], fontsize=5)
ax.set_ylim([-0.75, 17.75])
#plt.show()
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution2.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution2.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()      

dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_disease_chapters_extended.csv')
     

# plot 3
bthnames = np.asarray(['Alcohol', 'Smoking', 'High BP.', 'Low BP.', 'Height', 'Weight', 'Age at first Birth']).astype(object)
helpvar1 = sig_bth_cut * bthnames[:, None]
helpvar2 = sig_bth10 * bthnames[:, None]
a, b = np.unique(helpvar1, return_counts=True)
c, d = np.unique(helpvar2, return_counts=True)
a = a[1:]
b = b[1:]
c = c[1:]
d = d[1:]

dd = pd.DataFrame(np.concatenate((
            a[:, None],
            b[:, None],
            c[:, None],
            d[:, None]), axis=1))
dd.columns = ['bth_cut', 'count_cut', 'bth10', 'count10']
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_BTH.csv')

idx_sort = np.argsort(-b)
fig, ax = plt.subplots(1, 1, figsize=(1.5*cm,1.9*cm), dpi=600, sharex=True)
fig.subplots_adjust(hspace=0.05)
ax.barh(np.arange(7), b[idx_sort], lw=0.05, color='white', edgecolor='.3', hatch='xxxxxxxxxx')

ll = np.zeros((7, ))

for cc in range(22):
    helpvar1 = sig_bth_cut * bthnames[:, None]
    helpvar1 = helpvar1[:, cc] 
    a_, b = np.unique(helpvar1, return_counts=True)
    dd_ = pd.DataFrame(np.concatenate((
                a_[:, None],
                b[:, None]), axis=1))
    dd_.columns = ['bth_cut', 'count_cut_' + events[cc]]
    dd_ = dd_.iloc[1:, :]
    dd = dd.merge(dd_, how='left', on='bth_cut')
    dd.fillna(0, inplace=True)
    b = np.asarray(dd.iloc[:, -1])

    ax.barh(np.arange(7), b[idx_sort], left=ll[idx_sort], color='white', edgecolor=colormap[cc], hatch='xxxxxxxxxx', linewidth=0.5)
    ll += b


ax.plot(d[idx_sort], np.arange(7), color='black', ls='', marker='|', markersize=6)

ax.yaxis.tick_right()
ax.set_yticks(np.arange(7))
ax.set_yticklabels(np.asarray(a)[idx_sort], fontsize=5)
ax.set_ylim([-0.75, 6.75])

plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution3.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution3.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   


# plot 4
genenames = np.concatenate([np.repeat(Events[ii], 4).tolist() for ii in range(20)]).astype(object)
helpvar1 = sig_gene_cut * genenames[:, None]
helpvar2 = sig_gene10 * genenames[:, None]
a, b = np.unique(helpvar1, return_counts=True)
c, d = np.unique(helpvar2, return_counts=True)
a = a[1:]
b = b[1:]
c = c[1:]
d = d[1:]
            
dd = pd.DataFrame(pd.DataFrame(np.concatenate((
            a[:, None],
            b[:, None]), axis=1)))
dd1 = pd.DataFrame(pd.DataFrame(np.concatenate((
            c[:, None],
            d[:, None]), axis=1)))                
dd = dd.merge(dd1, left_on=0, right_on=0, how='left')
dd.fillna(0, inplace=True)
            
dd.columns = ['cancer_cut', 'count_cut', 'count10']
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_genealogy.csv')
  
a = dd.iloc[:, 0]
b = dd.iloc[:, 1]
c = dd.iloc[:, 0]
d = dd.iloc[:, 2]


idx_sort = np.argsort(-b)
fig, ax = plt.subplots(1, 1, figsize=(1.5*cm,4.55*cm), dpi=600, sharex=True)
fig.subplots_adjust(hspace=0.05)
ax.barh(np.arange(20), b[idx_sort], color='.3', lw=0.05, edgecolor='black', hatch='xxxxxxxxxxxx')
            
ll = np.zeros((20, ))

for cc in range(22):
    helpvar1 = sig_gene_cut * genenames[:, None]
    helpvar1 = helpvar1[:, cc] 
    a_, b = np.unique(helpvar1, return_counts=True)
    dd_ = pd.DataFrame(np.concatenate((
                a_[:, None],
                b[:, None]), axis=1))
    dd_.columns = ['cancer_cut', 'count_cut_' + events[cc]]
    dd_ = dd_.iloc[1:, :]
    dd = dd.merge(dd_, how='left', on='cancer_cut')
    dd.fillna(0, inplace=True)
    b = np.asarray(dd.iloc[:, -1])

    ax.barh(np.arange(20), b[idx_sort], left=ll[idx_sort], color=colormap[cc], lw=0.001, edgecolor='black', hatch='xxxxxxxxxxxx')
    ll += b
    
ax.plot(d[idx_sort], np.arange(20), color='white', ls='', marker='|', markersize=6)
ax.yaxis.tick_right()
ax.set_yticks(np.arange(20))
ax.set_yticklabels(np.asarray(a)[idx_sort], fontsize=5)
ax.set_xlabel('Count')
ax.set_ylim([-0.75, 19.75])
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution4.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution4.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   

dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_genealogy_extended.csv')
  

# plot 5
prop_dnpr1=[]
prop_gene1=[]
prop_bth1=[]

prop_dnpr2=[]
prop_gene2=[]
prop_bth2=[]

prop_dnpr3=[]
prop_gene3=[]
prop_bth3=[]

for cc in range(22):
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

# only necessary for ukb data - remove!!!!!!
prop_dnpr1 = np.zeros((22,))
prop_gene1 = np.zeros((22,))
prop_bth1 = np.zeros((22,))

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

#ax.set_yticklabels(['0', '0.5', '1])
ax.set_xlim([-0.75, 21.75])
ax.set_xticks(np.arange(22))
ax.set_xticklabels(np.asarray(Events), rotation=90)

handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[-1][-1], handles[-2][-1], handles[-3][-1]], labels[::-1], frameon=False, fontsize=5, bbox_to_anchor=(1.0, 0.75))

plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution5.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution5.pdf', dpi=600, bbox_inches='tight', transparent=True)
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
dd.columns = names
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_comparision.csv')


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

# only necessary for ukb data - remove!!!!!!
prop_dnpr1 = np.zeros((22,))
prop_gene1 = np.zeros((22,))
prop_bth1 = np.zeros((22,))
prop1 = np.zeros(22,)

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

plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution6.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/attribution6.pdf', dpi=600, bbox_inches='tight', transparent=True)
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
dd.columns = names
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/attribution_IQR.csv')



# %%
## Correlation
#======================================================================================================================
full = pred[:, 1, :20]
genealogy = pred2[:, :20, 1]
health = pred2[:, :20, 0] + pred2[:, :20, 2]
idxevents = pred[:, 0, :20].max(axis=1)==1

def corplot(dd, suffix='', sex=sex, idxevents=idxevents, sizemark=28, cmap=cmap, Events=Events, ROOT_DIR=ROOT_DIR, subset=True, corr=pearsonr):
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
    out.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/tables/corr_' + suffix + '.csv', sep=';')

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
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/corrplot_' + suffix + '.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/corrplot_' + suffix + '.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()  


# Main Figure
corplot(dd=full, suffix='full_pearsonr', corr=pearsonr)
corplot(dd=genealogy, suffix='genealogy_pearsonr', corr=pearsonr)  
corplot(dd=health, suffix='health_pearsonr', corr=pearsonr)  

corplot(dd=full, suffix='full_spearmanr', corr=spearmanr)
corplot(dd=genealogy, suffix='genealogy_spearmanr', corr=spearmanr)  
corplot(dd=health, suffix='health_spearmanr', corr=spearmanr)  

# %%
## Kaplan Meier Curves
#======================================================================================================================

quantile = 0.99
estll = []
for cc in range(22): #range(22):
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
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    lr1 = pp < np.quantile(pp, quantile)-10e-7 # weird 

    #cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=np.zeros_like(pred[idx, 1, :]), sex=sex.astype(int)[idx], full=False)
    #pp=[]
    #for ii in tqdm.tqdm(range(np.sum(idx))):
    #    pp.extend(cif_(ii))
    #pp = np.asarray(pp)
    #lr2 = pp < np.quantile(pp, 0.99)

    lr2 = A0_base[idx, cc] < np.quantile(A0_base[idx, cc], quantile)-10e-7 #weird

    tt_ = np.concatenate((tt_surv[idx][~lr1], tt_surv[idx][~lr2]))
    ee = np.concatenate((ee[~lr1], ee[~lr2]))
    lr = np.concatenate((np.zeros(((~lr1).sum())), np.ones(((~lr2).sum())))).astype(bool)

    dd = pd.DataFrame(np.concatenate((tt_[:, None], ee[:, None], (~lr[:, None]).astype(float)), axis=1))
    dd.columns = ['time', 'events', 'split']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/kmplots_raw.csv', sep=';')

    try:
        a = '''
        rm(list=ls())
        library(survival)
        ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/'
        '''

        b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/kmplots_raw.csv'"

        c = '''
        dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')

        m = coxph(Surv(time, events)~split, data=dd)

        x = paste(unname(exp(m$coefficients)), unname(exp(m$coefficients - 1.96*sqrt(m$var))), unname(exp(m$coefficients + 1.96*sqrt(m$var))), sep=';')

        write(x, file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_surv.txt', sep=''), append=FALSE, sep=";")

        surv_diff <- survdiff(Surv(time, events) ~ split, data = dd)
        x = paste(surv_diff$chisq, pchisq(surv_diff$chisq, 1, lower.tail=FALSE), sep=';')
        write(x, file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_logrank.txt', sep=''), append=FALSE, sep=";")

        '''

        with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_surv.R', 'w') as write_out:
            write_out.write(a+b+c)

        subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_surv.R'], shell=False)
        os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_surv.R')

        hr_est = np.loadtxt('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_surv.txt', delimiter=';')
        os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_surv.txt')

        logr_est = np.loadtxt('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_logrank.txt', delimiter=';')
        os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_logrank.txt')

        if np.abs(hr_est[0]) < 100000:
            hr_text = 'HR: ' + str(np.asarray(np.round(hr_est[0], 2))) + ' (' + str(np.asarray(np.round(hr_est[1], 2))) + ', ' + str(np.asarray(np.round(hr_est[2], 2))) + ')'
            if np.logical_or(np.logical_and(hr_est[1] > 1, hr_est[2] > 1), np.logical_and(hr_est[1] < 1, hr_est[2] < 1)):
                hr_text_small = '' + str(np.asarray(np.round(hr_est[0], 2))) + '*'
            else:
                hr_text_small = '' + str(np.asarray(np.round(hr_est[0], 2))) + ''
        else:
            hr_text = 'HR: -'
            hr_text_small = '-'
            
    except:
            hr_text = 'HR: -'
            hr_text = '-'
            logr_est = ['', '', '',  '']
            print('error')

    estll.extend([np.concatenate((hr_est, logr_est)).tolist()])

    fig, ax = plt.subplots(2, 1, figsize=(3.75*cm,4*cm), dpi=600, gridspec_kw={'height_ratios':[4, 1]})
    fig.subplots_adjust(wspace=0.0, hspace=0.4)

    times, km, ci, ll_1 = KM(tt_[lr], ee[lr], t1=[0, 365, 730, 1095], tmax=1196)
    m1 = np.min(ci[0])
    dd = pd.DataFrame(np.concatenate((times[:, None], km[:, None], ci[0][:, None], ci[1][:, None]), axis=1))
    dd.columns = ['times', 'km', 'lower95', 'upper95']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' +  events[cc] + '/data/kmplots1.csv', sep=';')

    ax[0].step(times, km, where='post', color='black',  lw=1)
    ax[0].step(times, ci[0], where='post', color='black', ls='--', lw=1, dashes=(1.5, 0.75))
    ax[0].step(times, ci[1], where='post', color='black', ls='--', lw=1, dashes=(1.5, 0.75))

    times, km, ci, ll_2 = KM(tt_[~lr], ee[~lr], t1=[0, 365, 730, 1095], tmax=1196)
    m2 = np.min(ci[0])
    dd = pd.DataFrame(np.concatenate((times[:, None], km[:, None], ci[0][:, None], ci[1][:, None]), axis=1))
    dd.columns = ['times', 'km', 'lower95', 'upper95']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' +  events[cc] + '/data/kmplots2.csv', sep=';')

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
    #plt.show()
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/km' +str(quantile)+'.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/km' +str(quantile)+'.pdf', dpi=600, bbox_inches='tight', transparent=True)
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
    
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/km_small.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/km_small.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

pd.DataFrame(estll).to_csv(ROOT_DIR + 'projects/CancerRisk/output/main/data/kmest' + str(quantile) + '.csv')


# %%
## Metrics
#=======================================================================================================================
concordance=[]
calibration = []
AUC=[]
Brier = []
predfrac = []
realfrac = []

for cc in range(22):
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
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    
    dd = pd.DataFrame(np.concatenate((tt_surv[idx, None], y_[:, None], pp[:, None]), axis=1).astype(float))
    dd.columns = ['time', 'events', 'split']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/metrics_raw.csv', sep=';')

    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/metrics_raw.csv'"

    c = '''
    dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')

    m = coxph(Surv(time, events)~split, data=dd)

    x = paste(unname(summary(m)$concordance[1]), unname(summary(m)$concordance[2]), sep=';')


    write(x, file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.txt', sep=''), append=FALSE, sep=";")
    '''

    with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.R'], shell=False)
    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.R')

    conc = np.squeeze(np.loadtxt('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.txt', delimiter=';'))
    concordance.extend([conc.tolist()])

    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.txt')
    
    a = '''
    rm(list=ls())
    ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/metrics_raw.csv'"

    c = '''
    dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')
    dd$odds <- log((dd$split/(1-dd$split)))
    m = glm(formula = events ~ odds, data = dd, family = binomial)

    x = paste(unname(m$coefficients[1]), unname(m$coefficients[2]), sep=';')
    write(x, file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_cali.txt', sep=''), append=FALSE, sep=";")
    '''

    with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_cali.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_cali.R'], shell=False)
    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_cali.R')

    cali = np.squeeze(np.loadtxt('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_cali.txt', delimiter=';'))
    calibration.extend([cali.tolist()])

    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_cali.txt')

    splits = 10
    step = np.ceil(np.sum(idx)/splits).astype(int)
    idxmat = np.zeros((np.sum(idx), splits))
    for kk in range(splits):
        idxmat[kk*step:(kk+1)*step, kk] = 1
    
    idxsort = np.argsort(pp)
    pp = pp[idxsort]
    y_ = y_[idxsort]
    tt_ = tt_[idxsort]
    
    #cases_pred = np.sum(idxmat * pp[:, None], axis=0)/idxmat.sum(axis=0)*100
    #cases_theo = np.sum(idxmat * y_[:, None], axis=0)/idxmat.sum(axis=0)*100
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
    cases_theo0 = np.asarray(cases_theo0)
    cases_theo1 = np.asarray(cases_theo1)
    cases_pred = np.asarray(cases_pred)
    predfrac.extend([cases_pred])
    realfrac.extend([cases_theo])
        
    #reg = LinearRegression().fit(cases_pred[:, None], cases_theo)
                       
    fpr, tpr, threshold = metrics.roc_curve(y_, pp)
    roc_auc = metrics.auc(fpr, tpr)
    
    #fpr_base, tpr_base, threshold_base = metrics.roc_curve(y_, pp_base)
    #roc_auc_base = metrics.auc(fpr_base, tpr_base)
    
    AUC.extend([roc_auc])

    Brier.extend([metrics.brier_score_loss(y_, pp)])
    #Matthews.extend([metrics.matthews_corrcoef(y_, pp)])
    #F1.extend([metrics.f1_score(y_, pp)])
    
    m = np.maximum(np.max([np.max(cases_theo)]), np.max([np.max(cases_pred)]))
    #txt = 'Intercept: ' + str(np.round(cali[0], 2)) + ', Slope: ' + str(np.round(cali[1], 2))
    fig, ax = plt.subplots(1, 2, figsize=(8.2*cm, 3.8*cm), dpi=600)
    fig.subplots_adjust(wspace=0.4, hspace=0.5)
        
    ax[1].plot([0, m], [0, m], color='black', ls='--', linewidth=0.5)
    ax[1].scatter(x=cases_pred, y=cases_theo, marker='x', s=8, linewidth=0.7, color=colormap[cc])
    
    
    ax[1].plot([cases_pred, cases_pred], [cases_theo0, cases_theo1], linewidth=0.7, color=colormap[cc])
    

    #ax[1].plot(np.asarray([0, m]), np.asarray([reg.intercept_, reg.intercept_+m*reg.coef_[0]]), color=colormap[cc], linewidth=1)
    ax[1].set_xlabel('Predicted Fraction in %')
    ax[1].set_ylabel('Realized Fraction in %')
    #ax[1].text(0.04, 0.007, txt, size=fontsize-1)
    

    #ax[0].plot(fpr_base, tpr_base, color='black', lw=1, label='Area: ' + str(np.round(roc_auc_base, 2)))                   
    ax[0].plot(fpr, tpr, color=colormap[cc], lw=1, label='Area: ' + str(np.round(roc_auc, 2)))
    ax[0].plot([0, 1], [0, 1], color='black', ls='--', lw=0.5)          
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlabel('False Positive Rate')
    ax[0].set_ylabel('True Positive Rate') 
    ax[0].legend(loc=4, prop={'size':fontsize-1}, frameon=False)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/metrics.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/metrics.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    dill.dump({'tpr':tpr,
               'fpr':fpr,
               'cases_pred':cases_pred,
               'cases_theo':cases_theo,
               'cases_theo0':cases_theo0,
               'cases_theo1':cases_theo1,
               'rocauc':roc_auc,
               'concordance':conc,
               }, open(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/metrics.pickle', 'wb'))
                                  
    #txt = str(np.round(cali[0], 2)) + ' / ' + str(np.round(cali[1], 2))
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

    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/metrics_small.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/metrics_small.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    
dd = pd.DataFrame(concordance)
dd.columns = ['concordance', 'se']
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/concordance.csv', sep=';')

# %%
## LR Test
#=======================================================================================================================
for cc in (range(22)):

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
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp.extend(cif_(ii))
    pp = np.asarray(pp)
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=pred2[:, :, (0, 2)].sum(axis=2), sex=sex.astype(int)[idx])
    pp_health=[]
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp_health.extend(cif_(ii))
    pp_health = np.asarray(pp_health)
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=pred2[idx, :, 1], sex=sex.astype(int)[idx])
    pp_gene=[]
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp_gene.extend(cif_(ii))
    pp_gene = np.asarray(pp_gene)
    
    cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1195, A0=A0, pred=np.zeros_like(pred[idx, 1, :]), sex=sex.astype(int)[idx], full=False)
    pp_base=[]
    for ii in tqdm.tqdm(range(np.sum(idx))):
        pp_base.extend(cif_(ii))
    pp_base = np.asarray(pp_base)
    
    pp_ukbest = A0_base[idx, cc]
    
    dd = pd.DataFrame(np.concatenate((tt_surv[idx, None], y_[:, None], pp[:, None], pp_health[:, None], pp_gene[:, None], pp_base[:, None], pp_ukbest[:, None], pred2[:, cc, (0, 2)].sum(axis=1)[idx, None], pred2[:, cc, 1][idx, None] ), axis=1).astype(float))

    dd.columns = ['time', 'events', 'all', 'health', 'gene', 'base', 'ukbest', 'r_health', 'r_gene']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/LR_raw.csv', sep=';')
    
    
    a = '''
    rm(list=ls())
    library(survival)
    ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/'
    '''

    b = 'data_name = ' + "'output/" + str(events[cc]) + "/data/LR_raw.csv'"

    c = '''
    dd <- read.csv(paste(ROOT_DIR, data_name, sep=''), header=TRUE, sep=';')


    m1 = coxph(Surv(time, events)~base, data=dd)
    m2 = coxph(Surv(time, events)~base+r_health, data=dd)
    m3 = coxph(Surv(time, events)~base+r_gene, data=dd)
    m4 = coxph(Surv(time, events)~base+r_gene+r_health, data=dd)
    m5 = coxph(Surv(time, events)~ukbest, data=dd)


    x = paste(m1$loglik[1], m1$loglik[2], m2$loglik[1], m2$loglik[2], m3$loglik[1], m3$loglik[2], m4$loglik[1], m4$loglik[2], m5$loglik[1], m5$loglik[2], sep=";")
    write(x, file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_LR.txt', sep=''), append=FALSE, sep=";")


    x = paste(unname(summary(m1)$concordance[1]), unname(summary(m1)$concordance[2]), unname(summary(m2)$concordance[1]), unname(summary(m2)$concordance[2]), unname(summary(m3)$concordance[1]), unname(summary(m3)$concordance[2]), unname(summary(m4)$concordance[1]), unname(summary(m4)$concordance[2]),
unname(summary(m5)$concordance[1]), unname(summary(m5)$concordance[2]),sep=';')

    write(x, file=paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.txt', sep=''), append=FALSE, sep=";")

    # ukbest vs all 
    p1 = pchisq(-2 * (m5$loglik[2] - m4$loglik[2]), 2, lower.tail=FALSE)

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

    p <- c(p1, p2, p3, p4, p5, p6)
    #res <- c()
    #for(padj in p.adjust.methods){res <- c(res, p.adjust(p, method=padj, n = length(p)))}
    dd = as.data.frame(matrix(p, ncol=6, byrow=TRUE))
    #row.names(dd) <- p.adjust.methods
    names(dd) <- c("ukbbest_all", "base_health", "base_gene", "all_health", "all_gene", "all_base")
    write.csv(dd, "/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_pvalues.csv")

    '''

    with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_LR.R', 'w') as write_out:
        write_out.write(a+b+c)

    subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_LR.R'], shell=False)
    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_LR.R')

    LR_est = np.squeeze(np.loadtxt('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_LR.txt', delimiter=';'))
    dd = pd.DataFrame(LR_est[None, :])


    dd.columns = ['logL_01', 'logL_base','logL_02', 'logL_base_health', 'logL_03', 'logL_base_gene', 'logL_04', 'logL_all', 'logL_05', 'logL_ukbest']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/LR.csv', sep=';')

    conc_est = np.squeeze(np.loadtxt('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.txt', delimiter=';'))
    dd = pd.DataFrame(conc_est[None, :])
    dd.columns = ['concordance_base', 'se_base','concordance_base_health', 'se_base_health', 'concordance_base_gene', 'se_base_gene', 'concordance_all', 'se_all', 'concordance_ukbest', 'se_ukbest']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/concordance_sub.csv', sep=';')

    pval_est = pd.read_csv('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_pvalues.csv')
    pval_est.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/pvalues.csv', sep=';')

    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_LR.txt')
    os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/est_conc.txt')
        
## PostProcessing
#=====================================================================================================================
dd1 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[0] + '/data/LR.csv', sep=';')
dd2 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[0] + '/data/pvalues.csv', sep=';')
dd3 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[0] + '/data/concordance_sub.csv', sep=';')

dd = pd.DataFrame(np.concatenate((

np.asarray(dd1.loc[:, ['Unnamed: 0', 'logL_base', 'logL_base_health', 'logL_base_gene', 'logL_all', 'logL_ukbest']]),
np.asarray(dd2.iloc[:, 2:]),
np.asarray(dd3.iloc[:, 1:])
), axis=1))

for cc in range(1, 22):
    dd1 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/LR.csv', sep=';')
    dd2 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/pvalues.csv', sep=';')
    dd3 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/concordance_sub.csv', sep=';')
    
    dd_helpvar = pd.DataFrame(np.concatenate((

    np.asarray(dd1.loc[:, ['Unnamed: 0', 'logL_base', 'logL_base_health', 'logL_base_gene', 'logL_all', 'logL_ukbest']]),
    np.asarray(dd2.iloc[:, 2:]),
    np.asarray(dd3.iloc[:, 1:])
    ), axis=1))

    dd = dd.append(dd_helpvar)

dd.columns = ['cancer', 'logL_base', 'logL_base_health', 'logL_base_gene', 'logL_all', 'logL_ukbest', 'pvalue_ukbbest_all', 'pvalue_base_health', 'pvalue_base_gene', 'pvalue_all_health', 'pvalue_all_gene', 'pvalue_all_base', 'concordance_base', 'se_base','concordance_base_health', 'se_base_health', 'concordance_base_gene', 'se_base_gene', 'concordance_all', 'se_all', 'concordance_ukbest', 'se_ukbest']

dd.iloc[:, 0] = Events

dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/tables/model_comparison.csv')

# %%
## Proportional Hazard
#======================================================================================================================
for cc in tqdm.tqdm(range(22)):
    dd = np.concatenate((age[:, None], (age+tt_surv)[:, None], pred[:, 0, cc, None], pred[:, 1, cc, None], sex[:, None]),  axis=1).astype(float)
    dd = pd.DataFrame(dd)
    dd.columns = ['start', 'stop', 'events', 'pred', 'sex']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/tmp/' + events[cc] + '_dd.csv', sep=';')
    

a = '''
rm(list=ls())
library(survival)

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/'
#ROOT_DIR = '/Users/alexwjung/Desktop/tmp/'


events = c('oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
            'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
            'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death')
res=c()
for(cc in seq(1, 22)){
print(cc)
dd <- read.csv(paste(ROOT_DIR, events[cc], '_dd.csv', sep=''), header=TRUE, sep=';')
dd$start = dd$start + dd$sex * 1000000
dd$stop = dd$stop + dd$sex * 1000000
m = coxph(Surv(start, stop, events)~pred, data=dd)

data_folder = paste('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/output/', events[cc], '/', sep='')

test.ph <- cox.zph(m)
write.csv(test.ph$table, paste(data_folder, 'tables/prophaz.csv', sep=''))
res = c(res, matrix(test.ph$table[1, ]))
}

write.csv(data.frame(matrix(res, ncol=3)), '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/output/main/tables/prophaz.csv')
'''

with open('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/test.R', 'w') as write_out:
    write_out.write(a)

subprocess.check_call(['Rscript', '/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/test.R'], shell=False)
os.remove('/nfs/research/sds/sds-ukb-cancer/projects/CancerRisk/tmp/test.R')

for cc in range(22):
    os.remove(ROOT_DIR + 'projects/CancerRisk/tmp/' + events[cc] + '_dd.csv')


# %%
## Relative Frequency
#=======================================================================================================================
for cc in tqdm.tqdm(range(22)):
    if cc in [7, 8, 9, 10]:
        idx = np.logical_and(~sex, out[:, cc]==1)
        idx2 = np.unique(np.concatenate([np.random.choice(np.where(np.logical_and(~idx, np.logical_and(np.logical_and(age >= age[ii] - 365, age < age[ii] + 365), sex==sex[ii])))[0], 100) for ii in np.where(idx)[0]]))
        
        dfre = np.zeros((1305,)) + 10e-100
        for ii in np.where(idx)[0]:
            dfre += dcodes[ii, :]
        dfre = dfre/np.where(idx)[0].shape[0]
        
        hfre = np.zeros((1305,)) + 10e-100
        for ii in idx2:
            hfre += dcodes[ii, :]
        hfre = hfre/idx2.shape[0]

        rr = np.round(dfre/hfre, 5)
        #rr[~idx_dfre] = 1
        #idx = np.logical_or(rr>1.01, rr<0.99)
        dd = pd.DataFrame(np.concatenate((disease_codes[:], rr[:, None], dfre[:, None], hfre[:, None], np.repeat(np.where(idx)[0].shape[0], 1305)[:, None], np.repeat(idx2.shape[0], 1305)[:, None]), axis=1))
        dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq', 'Cancer_count', 'Healthy_count']
        #dd.sort_values('RR', inplace=True)
        dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/relative_risk_matched.csv')

    elif cc in [11, 12]:
        idx = np.logical_and(sex,out[:, cc]==1)
        idx2 = np.unique(np.concatenate([np.random.choice(np.where(np.logical_and(~idx, np.logical_and(np.logical_and(age >= age[ii] - 365, age < age[ii] + 365), sex==sex[ii])))[0], 100) for ii in np.where(idx)[0]]))
        
        dfre = np.zeros((1305,)) + 10e-100
        for ii in np.where(idx)[0]:
            dfre += dcodes[ii, :]
        dfre = dfre/np.where(idx)[0].shape[0]
        
        hfre = np.zeros((1305,)) + 10e-100
        for ii in idx2:
            hfre += dcodes[ii, :]
        hfre = hfre/idx2.shape[0]

        rr = np.round(dfre/hfre, 5)
        #rr[~idx_dfre] = 1
        #idx = np.logical_or(rr>1.01, rr<0.99)
        dd = pd.DataFrame(np.concatenate((disease_codes[:], rr[:, None], dfre[:, None], hfre[:, None], np.repeat(np.where(idx)[0].shape[0], 1305)[:, None], np.repeat(idx2.shape[0], 1305)[:, None]), axis=1))
        dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq', 'Cancer_count', 'Healthy_count']
        #dd.sort_values('RR', inplace=True)
        dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/relative_risk_matched.csv')

    else:
        idx = out[:, cc]==1
        idx2 = np.unique(np.concatenate([np.random.choice(np.where(np.logical_and(~idx, np.logical_and(np.logical_and(age >= age[ii] - 365, age < age[ii] + 365), sex==sex[ii])))[0], 100) for ii in np.where(idx)[0]]))

        dfre = np.zeros((1305,)) + 10e-100
        for ii in np.where(idx)[0]:
            dfre += dcodes[ii, :]
        dfre = dfre/np.where(idx)[0].shape[0]
        
        hfre = np.zeros((1305,)) + 10e-100
        for ii in idx2:
            hfre += dcodes[ii, :]
        hfre = hfre/idx2.shape[0]

        rr = np.round(dfre/hfre, 5)
        #rr[~idx_dfre] = 1
        #idx = np.logical_or(rr>1.01, rr<0.99)
        dd = pd.DataFrame(np.concatenate((disease_codes[:], rr[:, None], dfre[:, None], hfre[:, None], np.repeat(np.where(idx)[0].shape[0], 1305)[:, None], np.repeat(idx2.shape[0], 1305)[:, None]), axis=1))
        dd.columns = ['icd10', 'disease', 'chapter', 'RR', 'Cancer_Freq', 'Healthy_Freq', 'Cancer_count', 'Healthy_count']
        #dd.sort_values('RR', inplace=True)
        dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/relative_risk_matched.csv')

    # Specific for Forest plot - 10% significance level
    dd = np.asarray(pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/relative_risk_matched.csv', usecols=[5, 6]))
    dfre = dd[:, 0]
    hfre = dd[:, 1]

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
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/forest_relative_risk_matched.csv')
    dd = pd.DataFrame(np.concatenate((disease_codes_sig[:, 0, None], rr_sig[:, None], dfre_sig[:, None], hfre_sig[:, None]), axis=1))
    dd.columns = ['icd10', 'RR', 'Cancer_Freq', 'Healthy_Freq']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/forest_relative_risk_matched_small.csv')

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


bthnames = np.asarray(['Alcohol', 'Smoking', 'High BP.', 'Low BP.', 'Height', 'Weight', 'Age at first Birth']).astype(object)
genenames = np.concatenate([np.repeat(Events[ii], 4).tolist() for ii in range(20)]).astype(object)

idx_dnpr = (sig_dnpr10.sum(axis=1) >3)
#idx_gene = (sig_gene10.sum(axis=1) >3) irrelevant
idx_bth = (sig_bth10.sum(axis=1) >3)

effect = np.exp(np.concatenate((theta_dnpr[idx_dnpr], theta_bth[idx_bth])))
names = np.concatenate((disease_codes[idx_dnpr, 1], bthnames[idx_bth]))

center_ = gmean(effect, axis=1)
idxsort = np.argsort(center_)

effect = effect[idxsort]
names = names[idxsort]
center_ = center_[idxsort]
min_ = effect.min(axis=1)
max_ = effect.max(axis=1)


# plot 2
dfre = []
hfre = []
for cc in tqdm.tqdm(range(20)):
    dd = np.asarray(pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/tables/relative_risk_matched.csv', usecols=[5, 6]))
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

pd.DataFrame(np.concatenate((names[:, None], dfre2, hfre2), axis=1)).to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/dfreq_matched.csv')
pd.DataFrame(np.concatenate((names[:, None], dfre2*event_count, hfre2*event_count), axis=1)).to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/dfreq_count_matched.csv')

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
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/disease_freq_matched.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/disease_freq_matched.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   


# %%
## Predictor Distribution
#=======================================================================================================================
for cc in tqdm.tqdm(range(22)):
    fig, ax = plt.subplots(1, 2, figsize=(7*cm, 3*cm), dpi=600, sharey=True, gridspec_kw={'width_ratios':[3, 1]})
    fig.subplots_adjust(wspace=0.05)
    ax[1].hist([pred[pred[:, 0, cc]==0, 1, cc], pred[pred[:, 0, cc]==1, 1, cc]], color=['.1', colormap[cc]], density=True, label=['Healthy', 'Cancer'], orientation="horizontal")
    ax[1].legend(frameon=False, fontsize=5)
    ax[1].set_xlabel('Freq.')
    idxcc = np.random.choice(np.where(pred[:, 0, cc]==0)[0], 50000)
    ax[0].plot((age/365)[idxcc], pred[idxcc, 1, cc], ls='', marker='x', markersize=0.75, markeredgewidth=0.05, color='.1')
    try:
        idxcc = np.random.choice(np.where(pred[:, 0, cc]==1)[0], 2500)
    except:
        idxcc = pred[:, 0, cc]==1
        
    ax[0].plot((age/365)[idxcc], pred[idxcc, 1, cc], ls='', marker='x', markersize=1, markeredgewidth=0.15, color=colormap[cc])
    ax[0].set_ylabel('Log(Hazard)')
    ax[0].set_xlabel('Age')
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/predictor_distribution.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/predictor_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

# %%
##  5yr Risk distribution
#=======================================================================================================================
for cc in (range(22)):
    ll = []
    for ii in tqdm.tqdm(range(377004)):
        ll.extend([np.asarray(Apred[ii])[:, cc].tolist()])

    ll = np.asarray(ll)

    qq_f = np.zeros((237, 9))
    qq_m = np.zeros((237, 9))
    for jj in range(237):
        
        if cc not in [11, 12]:
            idx_f = np.logical_and(ll[:, jj] > 0, ~sex)
            qq_f[jj, :] = np.quantile(ll[idx_f, jj], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        
        if cc not in [7, 8, 9, 10]:
            idx_m = np.logical_and(ll[:, jj] > 0, sex)
            qq_m[jj, :] = np.quantile(ll[idx_m, jj], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

    # plot1 
    fig, ax = plt.subplots(1, 2, figsize=(10*cm, 4*cm), dpi=600, sharey=True)
    fig.subplots_adjust(wspace=0.05)
    ax[0].plot(predage/365, qq_f[:, 0], color=colormap[cc], lw=1, ls=(0, (1, 1)), label='1st quantile')
    ax[0].plot(predage/365, qq_f[:, 1], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)), label='5th quantile')
    ax[0].plot(predage/365, qq_f[:, 2], color=colormap[cc], lw=1, ls=(0, (5, 1)), label='10th quantile')
    ax[0].plot(predage/365, qq_f[:, 3], color=colormap[cc], lw=1, ls='-', label='25th quantile')
    ax[0].plot(predage/365, qq_f[:, 4], color='black', lw=1, ls='-', label='median')
    ax[0].plot(predage/365, qq_f[:, 5], color=colormap[cc], lw=1, ls='-')
    ax[0].plot(predage/365, qq_f[:, 6], color=colormap[cc], lw=1, ls=(0, (5, 1)))
    ax[0].plot(predage/365, qq_f[:, 7], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)))
    ax[0].plot(predage/365, qq_f[:, 8], color=colormap[cc], lw=1, ls=(0, (1, 1)))

    ax[1].plot(predage/365, qq_m[:, 0], color=colormap[cc], lw=1, ls=(0, (1, 1)))
    ax[1].plot(predage/365, qq_m[:, 1], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)))
    ax[1].plot(predage/365, qq_m[:, 2], color=colormap[cc], lw=1, ls=(0, (5, 1)))
    ax[1].plot(predage/365, qq_m[:, 3], color=colormap[cc], lw=1, ls='-')
    ax[1].plot(predage/365, qq_m[:, 4], color='black', lw=1, ls='-')
    ax[1].plot(predage/365, qq_m[:, 5], color=colormap[cc], lw=1, ls='-')
    ax[1].plot(predage/365, qq_m[:, 6], color=colormap[cc], lw=1, ls=(0, (5, 1)))
    ax[1].plot(predage/365, qq_m[:, 7], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)))
    ax[1].plot(predage/365, qq_m[:, 8], color=colormap[cc], lw=1, ls=(0, (1, 1)))

    ax[0].set_ylabel('5-year Risk')
    
    if cc in [7, 8, 9, 10, 16]:
        ax[0].legend(frameon=False, fontsize=5)
    else:
        ax[0].legend(frameon=False, fontsize=5)

    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/5yr_risk_distribution.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/5yr_risk_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    dd = pd.DataFrame(np.concatenate((predage[:, None], qq_f, qq_m), axis=1))
    dd.columns =  ['age', 'q0.01_female', 'q0.05_female', 'q0.1_female', 'q0.25_female', 'q0.5_female', 'q0.75_female', 'q0.9_female', 'q0.95_female', 'q0.99_female', 'q0.01_male', 'q0.05_male', 'q0.1_male', 'q0.25_male', 'q0.5_male', 'q0.75_male', 'q0.9_male', 'q0.95_male', 'q0.99_male']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/5yr_risk_distribution.csv')

# %%
## Absolute Risk Distribution
#=======================================================================================================================#
# Cancer specific plot
for cc in range(21, 22):
    if cc in [7, 8, 9, 10]:
        idx = ~sex
    elif cc in [11, 12]:
        idx = sex
    else:
        idx = np.logical_or(sex, ~sex)
    
    qq = np.quantile(prediction[idx, cc], 0.99)
    fig, ax = plt.subplots(1, 2, figsize=(7*cm, 3*cm), dpi=300, gridspec_kw={'width_ratios':[4, 1]}, sharey=True)
    fig.subplots_adjust(wspace=0.1)
    
    idxss = np.random.choice(np.where(np.logical_and(~out[:, cc].astype(bool), idx))[0], 50000)
    ax[0].plot(age[idxss]/365, np.minimum(prediction[idxss, cc], qq), ls='', marker='x', markersize=0.75, markeredgewidth=0.05, color='.1')

    try:
        idxss = np.random.choice(np.where(np.logical_and(out[:, cc].astype(bool), idx))[0], 2500)
    except:
        idxss = np.logical_and(out[:, cc].astype(bool), idx)
    ax[0].plot(age[idxss]/365, np.minimum(prediction[idxss, cc], qq), ls='', marker='x', markersize=1, markeredgewidth=0.25, color=colormap[cc])

    ax[1].hist([np.minimum(prediction[idx, cc][~out[idx, cc].astype(bool)], qq), np.minimum(prediction[idx, cc][out[idx, cc].astype(bool)], qq)], color=['0.1', colormap[cc]], density=True, orientation="horizontal", bins=10, label=['Non-Cancer', 'Cancer'])

    ax[0].set_ylabel('Absolute Risk 5yr.')
    ax[0].set_xlabel('Age')
    ax[0].set_xticks([50, 60, 70])
    #ax[0].set_ylim([0, 0.4])
    ax[1].set_xlabel('Frequency')
    #ax[1].legend(frameon=False, fontsize=5, bbox_to_anchor=(0.4, 0.85))
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/absolute_risk_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/absolute_risk_distribution.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    

# %%
## Screening
#=======================================================================================================================#
## Data Table - Additional
for cc in tqdm.tqdm(range(20)):
    aa_ = []
    base_n = [] 
    base_c = []
    base_age_mean = []
    base_age_se = []
    base_age_q10 = []
    base_age_q25 = []
    base_age_median = []
    base_age_q75 = []
    base_age_q90 = []

    phs_n = []
    phs_c = []
    phs_age_mean = []
    phs_age_se = []
    phs_age_q10 = []
    phs_age_q25 = []
    phs_age_median = []
    phs_age_q75 = []
    phs_age_q90 = []
    
    for aa in range(55, 66):
        aa_.extend([aa])
        # PPV for aa - aa+1 year range
        idxage_cut = np.logical_and(age/365 >= aa, age/365 < aa+5)
        idxage_early = age/365 < aa+5

        idx_early_female = np.logical_and(~sex, idxage_early)
        idx_early_male = np.logical_and(sex, idxage_early)

        n_f = (~sex[idxage_cut]).sum()
        n_m = sex[idxage_cut].sum()

        target_f = np.sort(prediction[idx_early_female, cc])[-n_f]
        target_m = np.sort(prediction[idx_early_male, cc])[-n_m]

        screen = np.zeros((prediction.shape[0],))

        if cc in [7, 8, 9, 10]:
            screen[np.logical_and(idx_early_female, prediction[:, cc] >= target_f)] = 1
            idxage_cut = np.logical_and(~sex, idxage_cut)
        elif cc in [11, 12]:
            screen[np.logical_and(idx_early_male, prediction[:, cc] >= target_m)] = 1
            idxage_cut = np.logical_and(sex, idxage_cut)
        else:
            screen[np.logical_and(idx_early_female, prediction[:, cc] >= target_f)] = 1
            screen[np.logical_and(idx_early_male, prediction[:, cc] >= target_m)] = 1

        screen = screen.astype(bool)

        #bdd1.extend([age[np.logical_and(idxage_cut, out[:, cc]==1)]/365]) 
        #bdd2.extend([age[np.logical_and(screen, out[:, cc]==1)]/365]) 

        base_n.extend([idxage_cut.sum()])
        base_c.extend([np.logical_and(idxage_cut, out[:, cc]==1).sum()])
        base_age_mean.extend([np.mean(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365)])
        base_age_se.extend([np.sqrt(np.var(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365))])
        base_age_q10.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.1)])
        base_age_q25.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.25)])
        base_age_median.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.5)])
        base_age_q75.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.75)])
        base_age_q90.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.9)])

        phs_n.extend([screen.sum()])
        phs_c.extend([np.logical_and(screen, out[:, cc]==1).sum()])
        phs_age_mean.extend([np.mean(age[np.logical_and(screen, out[:, cc]==1)]/365)])
        phs_age_se.extend([np.sqrt(np.var(age[np.logical_and(screen, out[:, cc]==1)]/365))])
        phs_age_q10.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.1)])
        phs_age_q25.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.25)])
        phs_age_median.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.5)])
        phs_age_q75.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.75)])
        phs_age_q90.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.9)])

    aa_ = np.asarray(aa_)
    base_n = np.asarray(base_n)
    base_c = np.asarray(base_c)
    base_age_mean = np.asarray(base_age_mean)
    base_age_se = np.asarray(base_age_se)
    base_age_q10 = np.asarray(base_age_q10)
    base_age_q25 = np.asarray(base_age_q25)
    base_age_median = np.asarray(base_age_median)
    base_age_q75 = np.asarray(base_age_q75)
    base_age_q90 = np.asarray(base_age_q90)
    phs_n = np.asarray(phs_n)
    phs_c = np.asarray(phs_c)
    phs_age_mean = np.asarray(phs_age_mean)
    phs_age_se = np.asarray(phs_age_se)
    phs_age_q10 = np.asarray(phs_age_q10)
    phs_age_q25 = np.asarray(phs_age_q25)
    phs_age_median = np.asarray(phs_age_median)
    phs_age_q75 = np.asarray(phs_age_q75)
    phs_age_q90 = np.asarray(phs_age_q90)

    dd = pd.DataFrame(np.concatenate((aa_[:, None],
        base_n[:, None],
        base_c[:, None],
        base_age_mean[:, None],
        base_age_se[:, None],
        base_age_q10[:, None],
        base_age_q25[:, None],
        base_age_median[:, None],
        base_age_q75[:, None],
        base_age_q90[:, None],
        phs_n[:, None],
        phs_c[:, None],
        phs_age_mean[:, None],
        phs_age_se[:, None],
        phs_age_q10[:, None],
        phs_age_q25[:, None],
        phs_age_median[:, None],
        phs_age_q75[:, None],
        phs_age_q90[:, None]
    ), axis=1))
    dd.columns = ['age_threshold', 'base_n', 'base_c', 'base_age_mean', 'base_age_se', 'base_age_q10', 'base_age_q25', 'base_age_median', 'base_age_q75', 'base_age_q90', 'phs_n', 'phs_c', 'phs_age_mean', 'phs_age_se', 'phs_age_q10', 'phs_age_q25', 'phs_age_median', 'phs_age_q75', 'phs_age_q90', ]
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/screening.csv')

    
    
## Figure
bdd1 = []
bdd2 = []
age_q25 = []
age_m = []
rr = []
phs_cases = []
baseline_cases = []
n = []

aa = 55
for cc in tqdm.tqdm(range(20)):
    if cc not in [6, 8, 12, 16]:
        # PPV for aa - aa+1 year range
        idxage_cut = np.logical_and(age/365 >= aa, age/365 < aa+5)
        idxage_early = age/365 < aa+5

        idx_early_female = np.logical_and(~sex, idxage_early)
        idx_early_male = np.logical_and(sex, idxage_early)

        n_f = (~sex[idxage_cut]).sum()
        n_m = sex[idxage_cut].sum()

        target_f = np.sort(prediction[idx_early_female, cc])[-n_f]
        target_m = np.sort(prediction[idx_early_male, cc])[-n_m]

        screen = np.zeros((prediction.shape[0],))

        if cc in [7, 8, 9, 10]:
            screen[np.logical_and(idx_early_female, prediction[:, cc] >= target_f)] = 1
            idxage_cut = np.logical_and(~sex, idxage_cut)
        elif cc in [11, 12]:
            screen[np.logical_and(idx_early_male, prediction[:, cc] >= target_m)] = 1
            idxage_cut = np.logical_and(sex, idxage_cut)
        else:
            screen[np.logical_and(idx_early_female, prediction[:, cc] >= target_f)] = 1
            screen[np.logical_and(idx_early_male, prediction[:, cc] >= target_m)] = 1

        screen = screen.astype(bool)
        
        phs_cases.extend([(out[screen, cc]==1).sum()])
        baseline_cases.extend([(out[idxage_cut, cc]==1).sum()])
        n.extend([idxage_cut.sum()])

        bdd1.extend([age[np.logical_and(idxage_cut, out[:, cc]==1)]/365]) 
        bdd2.extend([age[np.logical_and(screen, out[:, cc]==1)]/365]) 
        rr.extend([np.round(np.logical_and(screen, out[:, cc]==1).sum()/np.logical_and(idxage_cut, out[:, cc]==1).sum(), 2)])
        age_q25.extend([np.round(np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.25) - np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.25) , 2)])
        age_m.extend([np.round(np.mean(age[np.logical_and(screen, out[:, cc]==1)]/365) - np.mean(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365) , 2)])
        
        print(Events[cc], np.round(np.logical_and(screen, out[:, cc]==1).sum()/np.logical_and(idxage_cut, out[:, cc]==1).sum(), 2))
        print(np.round(np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.25) - np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.25) , 2))
        print(np.round(np.mean(age[np.logical_and(screen, out[:, cc]==1)]/365) - np.mean(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365) , 2))
    
       
Events_ = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Breast', 
                 'Corpus Uteri', 'Ovary', 'Prostate', 'Kidney', 'Bladder', 'Brain',
                'NHL', 'MM', 'AML']

colormap_= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#6e0b3c', 
                 '#7A378B', '#CD6090', '#006400', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CD6600', '#FF8C69', '#8f0000'])

fig, ax = plt.subplots(1, 1, figsize=(15*cm, 3*cm), dpi=300)

bplot1 = ax.boxplot(bdd1, notch=False,
                     vert=True, showfliers=False,
                     patch_artist=True, medianprops=dict(color='black', lw=1), boxprops=dict(lw=0.3, hatch='xxxxxxxx'), whiskerprops=dict(lw=0.5), positions=np.arange(16), widths=0.35,  capprops=dict(lw=0.5))


bplot2 = ax.boxplot(bdd2, notch=False,
                     vert=True, showfliers=False,
                     patch_artist=True, medianprops=dict(color='black', lw=1), boxprops=dict(lw=0.3), whiskerprops=dict(lw=0.5), positions=np.arange(16)+0.4, widths=0.35,  capprops=dict(lw=0.5))

# fill with colors
for patch, color in zip(bplot1['boxes'], colormap_):
    patch.set_facecolor(color)

for patch, color in zip(bplot2['boxes'], colormap_):
    patch.set_facecolor(color)

    
#ax.set_xlim([0.5, 20.5])
ax.set_xticks(np.arange(0, 16)+0.2)
ax.set_xticklabels(np.asarray(Events_), rotation=90)

ax.set_yticks([50, 55, 60])

ax.axhline(60, color='black', lw=0.75)
ax.axhline(55, color='black', lw=0.75)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/screening.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close() 

# figure 2

phs_cases = np.asarray(phs_cases)
baseline_cases = np.asarray(baseline_cases)
n = np.asarray(n)
rr = phs_cases/baseline_cases
rr_l, rr_u = np.exp(np.log(phs_cases/baseline_cases) - np.sqrt((n-phs_cases)/phs_cases/n + (n-baseline_cases)/baseline_cases/n)), np.exp(np.log(phs_cases/baseline_cases) + np.sqrt((n-phs_cases)/phs_cases/n + (n-baseline_cases)/baseline_cases/n))

mpl.rcParams['axes.spines.bottom'] = False
fig, ax = plt.subplots(1, 1, figsize=(15*cm, 1*cm), dpi=300)

ax.bar(range(16), rr-1, color=colormap_, bottom=1, width=0.5)

for cc in range(16):
    ax.plot([cc, cc], [rr_l[cc], rr_u[cc]], lw=0.75,  color='black')

ax.axhline(1, lw=0.5, color='black')
#ax.set_ylim([0.5, 1.5])
ax.set_xlim([-0.75, 15.6])
ax.set_xticks(np.arange(0, 16))
ax.set_xticklabels(np.asarray(Events_), rotation=90)
ax.get_xaxis().set_visible(False)

plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/screening2.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close() 
mpl.rcParams['axes.spines.bottom'] = True

# Summary Table
dd = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[0] + '/data/screening.csv')
dd = dd.iloc[[0, 5, 10], :]
dd.reset_index(inplace=True, drop=True)
dd.iloc[:, 0] = np.repeat(Events[0], 3)

for cc in range(1, 20):
    dd_ = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/data/screening.csv')
    dd_ = dd_.iloc[[0, 5, 10], :]
    dd_.reset_index(inplace=True, drop=True)
    dd_.iloc[:, 0] = np.repeat(Events[cc], 3)
    dd = dd.append(dd_)
    
dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/main/tables/screening.csv')



# %%
## Concordance
#=======================================================================================================================

concordance=[]
for cc in tqdm.tqdm(range(22)):
    dd = np.concatenate((age[:, None], (age+tt_surv)[:, None], pred[:, 0, cc, None], pred[:, 1, cc, None], sex[:, None]),  axis=1).astype(float)
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




conc = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/concordance.csv', sep=';')
conc2 = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/data/concordance2.csv', sep=';')
concordance = np.asarray(conc.iloc[:, 1:])
concordance2 = np.asarray(conc2.iloc[:, -2:])
ll = [round_(concordance[ii, 0], 2) + ' (' + round_(concordance[ii, 1], 2) + ')' + '\n' + round_(concordance2[ii, 0], 2) + ' (' + round_(concordance2[ii, 1], 2) + ')' for ii in range(22)]

# plot 1
fig, ax = plt.subplots(1, 1, figsize=(3.3*cm, 15*cm), dpi=600)
ax.barh(width=concordance[:, 0], y=np.arange(22)+0.22, color=colormap, edgecolor=colormap ,lw=0.5, height=0.4, xerr=concordance[:, 1])
ax.barh(width=concordance2[:, 0], y=np.arange(22)-0.22, color='white', edgecolor=colormap ,lw=0.5, hatch='xxxxxxxxxxxx', height=0.4, xerr=concordance2[:, 1])

#ax.plot(concordance[:, 6], np.arange(22), ls='', marker='|', color='black', ms=10)
ax.set_yticks(np.arange(22))
ax.set_yticklabels(Events)
ax.set_xlim(0.5, 0.85)
ax.set_xticks([0.5, 0.6, 0.7, 0.8])
ax.set_xticklabels(['0.5', '', '', '0.8'])
ax.set_ylim([-0.8, 21.8])
ax0 = ax.twinx()
ax0.set_ylim(ax.get_ylim())
ax0.set_yticks(np.arange(22))

ax0.set_yticklabels(ll)
ax.set_xlabel('Concordance')

#plt.show()
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/concordance_summary.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/concordance_summary.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# plot 2
concordance_den = np.asarray([0.75, 0.70, 0.67, 0.81, 0.69, 0.73, 0.60, 0.57, 0.61, 0.68, 0.63, 0.71, 0.59, 0.67, 0.75, 0.63, 0.66, 0.65, 0.67, 0.71, 0.64, 0.82])
concordanceSE_den = np.asarray([0.006, 0.007, 0.003, 0.007, 0.006, 0.003, 0.005, 0.003, 0.014, 0.007, 0.009, 0.002, 0.02, 0.006, 0.006, 0.009, 0.013, 0.006, 0.009, 0.015, 0.002, 0.001])
concordance_N = np.asarray([282,220,1446,189,352,1077,652,2217,32,371,265,2271,11,359,611,209,94,526,246,56,5940,3374])

fig, ax = plt.subplots(1, 1, figsize=(5*cm, 5*cm), dpi=200)
ax.plot([0.5, 1], [0.5, 1], color='black', ls='--', lw=0.5)


# v0
#ax.scatter(concordance_den, concordance[:, 0], color=colormap, s=8) 

# v1
for cc in range(22):
    ax.scatter(concordance_den[cc], concordance[cc, 0], color=colormap[cc], s=6) 
    ax.plot([concordance_den[cc], concordance_den[cc]], [concordance[cc, 0]-concordance[cc, 1], concordance[cc, 0]+concordance[cc, 1]], color=colormap[cc], lw=1) #v1
    ax.plot([concordance_den[cc]-concordanceSE_den[cc], concordance_den[cc]+concordanceSE_den[cc]], [concordance[cc, 0], concordance[cc, 0]], color=colormap[cc], lw=1) #v1

# v2
#ax.scatter(concordance_den, concordance[:, 0], color=colormap, s=np.log(concordance_N/concordance_N.max()*1000), lw=0.1) 


#ax.scatter(concordance_den, concordance[:, 0], color=colormap, s=(concordance_N/concordance_N.max())*100, lw=0.1) 


ax.plot([0.5, 1], [0.5, 1], color='black', ls='--', lw=0.5)

ax.set_xlim([0.5, 0.85])
ax.set_xticks([0.5, 0.6, 0.7, 0.8])
ax.set_xticklabels(['', '0.6', '', '0.8'])

ax.set_ylim([0.5, 0.85])
ax.set_yticks([0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels(['', '0.6', '', '0.8'])
#ax.set_ylabel('UKB')
#ax.set_xlabel('Denmark')


plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/main/figures/concordance1.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

print('R2: ', r2_score(concordance_den, concordance[:, 0]))
print('pearsonR: ', pearsonr(concordance_den, concordance[:, 0]))

# plot 3
concordance_den = np.asarray([0.63534127,0.00846353,
0.56899393,0.00804525,
0.54500303,0.00289135,
0.73601542,0.00905208,
0.5741357,0.00681678,
0.66510997,0.00309185,
0.5821417,0.00393321,
0.55202766,0.0028199,
0.59039945,0.00866579,
0.63874004,0.00697881,
0.55170481,0.00850181,
0.56612478,0.00311738,
0.58027424,0.00860602,
0.59190729,0.00676117,
0.57040981,0.00771835,
0.53970235,0.00856643,
0.59450331,0.00973591,
0.54968275,0.0059916,
0.53868702,0.00976402,
0.54295919,0.01562163,
0.55616084,0.00176661,
0.76322943,0.00146905])
concordance_den = np.reshape(concordance_den, (22, 2), 'C')
concordance_N = np.asarray([282,220,1446,189,352,1077,652,2217,32,371,265,2271,11,359,611,209,94,526,246,56,5940,3374])


fig, ax = plt.subplots(1, 1, figsize=(5*cm, 5*cm), dpi=300)

#v0
#ax.scatter(concordance_den[:, 0], concordance2[:, 0], color=colormap, s=8)

# v1
for cc in range(22):
    ax.scatter(concordance_den[cc, 0], concordance2[cc, 0], color=colormap[cc], s=6) 
    ax.plot([concordance_den[cc, 0], concordance_den[cc, 0]], [concordance2[cc, 0]-concordance2[cc, 1], concordance2[cc, 0]+concordance2[cc, 1]], color=colormap[cc], lw=1) #v1
    ax.plot([concordance_den[cc,0]-concordance_den[cc, 1], concordance_den[cc,0]+concordance_den[cc, 1]], [concordance2[cc, 0], concordance2[cc, 0]], color=colormap[cc], lw=1) #v1


#v2
#ax.scatter(concordance_den[:, 0], concordance2[:, 0], color=colormap, s=np.log(concordance_N/concordance_N.max()*5000), lw=0.1)
#ax.scatter(concordance_den[:, 0], concordance2[:, 0], color=colormap, s=np.log(concordance_N/concordance_N.max()*1000), lw=0.1)


ax.plot([0.5, 1], [0.5, 1], color='black', ls='--', lw=0.5)
ax.set_xlim([0.5, 0.85])
ax.set_xticks([0.5, 0.6, 0.7, 0.8])
ax.set_xticklabels(['', '0.6', '', '0.8'])

ax.set_ylim([0.5, 0.85])
ax.set_yticks([0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels(['', '0.6', '', '0.8'])
#ax.set_ylabel('UKB')
#ax.set_xlabel('Denmark')


plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/main/figures/concordance2.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

print('R2: ', r2_score(concordance_den[:, 0], concordance2[:, 0]))
print('pearsonR: ', pearsonr(concordance_den[:, 0], concordance2[:, 0]))

# %%


# %%
print('finsihed')

# %%
exit()

# %%
dtype = torch.FloatTensor 

# %%
data = [[], [], [], []]

# %%
# Patient X 
# ==========================================================================================
# ==========================================================================================
time = np.round(np.asarray([20.745, 24.7623, 27.5467, 29.40, 30.048, 31.6723, 35.12, 40.5387, 45.587, 51.23498, 55.23478, 57.2364, 60.8723]) * 365)
dnpr_dummies = np.zeros((time.shape[0], 1305))
genealogy = np.ones((time.shape[0], 80)) * -1
bth = np.zeros((time.shape[0], 7))

# inital imputation for bth
bth[:, -2] = -0.01 # weight
bth[:, -3] = -0.08 # height 
bth[4:, -1] = time[4]/365/100 # age at first brith

# set cases
dnpr_dummies[1:, 984]=1

dnpr_dummies[2:, 771]=1
genealogy[2:, 5]=1 

dnpr_dummies[3:, 1009]=1
dnpr_dummies[3:, 1012]=1

dnpr_dummies[4:, 1049]=1

dnpr_dummies[5:, 973]=1
dnpr_dummies[5:, 655]=1

bth[6:, 0] = 1
bth[6:, 1] = -1
bth[6:, 2] = 1
bth[6:, 3] = -1

dnpr_dummies[7:, 733]=1
genealogy[7:, 29]=1 
genealogy[7:, 9]=1 
genealogy[7:, 11]=1 

bth[8:, -2] = 0.13
dnpr_dummies[8:, 307]=1
dnpr_dummies[8:, 706]=1

dnpr_dummies[9:, 264]=1
dnpr_dummies[9:, 601]=1
dnpr_dummies[9:, 702]=1

genealogy[10:, 28]=1 
genealogy[10:, 30]=1 

dnpr_dummies[11:, 960]=1

dnpr_dummies[12:, 740]=1

data[0] = torch.tensor(data[0]).type(dtype)
data[1] = torch.from_numpy(dnpr_dummies).type(dtype)
data[2] = torch.from_numpy(genealogy).type(dtype)
data[3] = torch.from_numpy(bth).type(dtype)

# %%


# %%


# %%
    theta_dnpr_lower, theta_dnpr, theta_dnpr_upper  = gg.quantiles([0.05, 0.5, 0.95])['theta_dnpr']
    theta_dnpr_lower = theta_dnpr_lower.detach().numpy()

# %%
# Prediction 
# ==========================================================================================
# ==========================================================================================
pp=[]
for cc in tqdm.tqdm(range(len(tt))):
    pyro.clear_param_store()
    with torch.no_grad(): 
        #tt = pickle.load(open(dir_out + 'm1/model/' + events[cc] + '/param.pkl', 'rb'))
        mm = tt[cc]['model']        
        gg = tt[cc]['guide']

        theta_dnpr = gg.quantiles([0.5])['theta_dnpr'][0].detach().numpy()
        theta_gene = gg.quantiles([0.5])['theta_gene'][0].detach().numpy()
        theta_bth = gg.quantiles([0.5])['theta_bth'][0].detach().numpy()

    
        p_dnpr = np.matmul(dnpr_dummies, theta_dnpr.T)
        p_gene = np.matmul(genealogy, theta_gene.T)
        p_bth = np.matmul(bth, theta_bth.T)
        pred = p_dnpr + p_gene + p_bth 
        pp.extend([pred])
pp= np.squeeze(np.stack(pp)).T

# %%
# Prediction  2
# ==========================================================================================
# ==========================================================================================
time_int = np.concatenate((np.concatenate((np.asarray([0]), time))[:-1, None], 
time[:, None]), axis=1)
pred = pp.copy()
risk = []
for age in np.arange(1, 60*365, 1):
    idx = np.logical_and(age <= time_int[:, 1], age > time_int[:, 0])
    rr = []
    for cc in range(22):
        cif_ = CIF(cc=cc, tt0=np.asarray([age]), tt_range=1825, A0=A0, pred=pred[idx, :], sex=np.asarray([0]), full=False)
        rr.extend(cif_(0))
    risk.extend([rr])
risk = np.asarray(risk)
ymax=np.asarray(risk).sum(axis=1)[(np.floor(time_int[:, 1]/365).astype(int)-20)]

# base
risk2 = []
for age in np.arange(1, 60*365, 1):
    idx = np.logical_and(age <= time_int[:, 1], age > time_int[:, 0])
    rr = []
    for cc in range(22):
        cif_ = CIF(cc=cc, tt0=np.asarray([age]), tt_range=1825, A0=A0, pred=np.zeros_like(pred[idx, :]), sex=np.asarray([0]), full=False)
        rr.extend(cif_(0))
    risk2.extend([rr])
risk2 = np.asarray(risk2)

# %%
# Schematic
# ==========================================================================================
# ==========================================================================================
fig, ax = plt.subplots(1, 1, figsize=(5*cm, 3*cm), sharex=True, dpi=600)
ax.set_ylim([0, 0.25])
ymax = np.asarray(risk).sum(axis=1)
for ii in range(12):
    ax.axvline(x=np.floor(time_int[ii, 1]), ymax=ymax[int(time_int[ii, 1])]*4,  color='black', ls='--', linewidth=0.5)
ax.step(range(1, 60*365, 1), np.asarray(risk2).sum(axis=1), color='.2', linewidth=0.7)
ax.step(range(1, 60*365, 1), np.asarray(risk).sum(axis=1), color='red', linewidth=0.85)

ax.set_ylim([0, 0.25])
ax.set_yticks([0.00, 0.1, 0.20])
ax.set_xlim([15*365, 60*365])
ax.set_xticks([20*365, 40*365, 60*365])
ax.set_xticklabels([20, 40, 60])
ax.set_ylabel('CIF(t)')
ax.set_xlabel('Age')

plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + 'main' + '/figures/schematic.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# %%



