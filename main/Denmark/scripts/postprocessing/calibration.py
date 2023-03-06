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

#from statsmodels.stats.multitest import multipletests
from scipy.stats import spearmanr
from scipy.stats import pearsonr

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = '/Users/alexwjung/Desktop/ee/'


events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Melanoma', 'Breast', 
                'Cervix Uteri', 'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney', 'Bladder', 'Brain',
                'Thyroid', 'NHL', 'MM', 'AML', 'Other', 'Death']

for cc in range(22):
    print(cc, events[cc])

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

cp = []
ct = []

for cc in range(22):
    
    cp.extend([dill.load(open(ROOT_DIR + 'danish_extract/' + events[cc] + '/data/metrics.pickle', 'rb'))['cases_pred']])
    ct.extend([dill.load(open(ROOT_DIR + 'danish_extract/' + events[cc] + '/data/metrics.pickle', 'rb'))['cases_theo']])  
    
cp = np.asarray(cp)/100
ct = np.asarray(ct)/100

fig, ax = plt.subplots(1, 1, figsize=(6*cm, 5*cm), dpi=300)
for cc in range(20):
    ax.scatter(cp[cc, :], ct[cc, :], color=colormap[cc], s=4) 

ax.set_ylim([0.0000005, 0.1])
ax.set_xlim([0.0000005, 0.1])

ax.plot(ax.get_xlim(), ax.get_ylim(), color='black', ls='--', lw=0.5)

#ax.set_ylabel('Realized %')
#ax.set_xlabel('Predicted %')
#ax.set_xticks([0.0000001,0.000001,0.00001,0.0001,0.001, 0.1])
#ax.set_yticks([0.0000001,0.000001,0.00001,0.0001,0.001, 0.1])
ax.set_yscale('log')
ax.set_xscale('log')

plt.savefig(ROOT_DIR + 'calibration_denmark.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# %%
ROOT_DIR + 'ukb_extract/' + events[cc] + '/data/metrics.pickle'
cp = []
ct = []

for cc in range(22):
    
    cp.extend([dill.load(open(ROOT_DIR + 'ukb_extract/' + events[cc] + '/data/metrics.pickle', 'rb'))['cases_pred']])
    ct.extend([dill.load(open(ROOT_DIR + 'ukb_extract/' + events[cc] + '/data/metrics.pickle', 'rb'))['cases_theo']])  
    
cp = np.asarray(cp)/100
ct = np.asarray(ct)/100

fig, ax = plt.subplots(1, 1, figsize=(6*cm, 5*cm), dpi=200)
for cc in range(20):
    ax.scatter(cp[cc, :], ct[cc, :], color=colormap[cc], s=3) 

ax.set_ylim([0.00001, 0.1])
ax.set_xlim([0.00001, 0.1])

ax.plot(ax.get_xlim(), ax.get_ylim(), color='black', ls='--', lw=0.5)


#ax.set_ylabel('Realized %')
#ax.set_xlabel('Predicted %')
ax.set_yscale('log')
ax.set_xscale('log')

plt.savefig(ROOT_DIR + 'calibration_ukb.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

# %%


# %%


# %%


# %%



