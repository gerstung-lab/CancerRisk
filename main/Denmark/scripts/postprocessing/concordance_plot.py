# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import pickle
import tqdm

import pandas as pd
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

import warnings
warnings.filterwarnings("ignore")

# model - ProbCox
sys.path.append('/users/projects/cancer_risk/main/scripts/ProbCox')
import probcox as pcox 

# dataloader
sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from dataloader import Data_Pipeline

# help functions
from helper import custom_collate, IterativeSampler, RandomSampler

# seeding
np.random.seed(11)
torch.random.manual_seed(12)
pyro.set_rng_seed(13)

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'
dir_out = '/users/projects/cancer_risk/main/output/'

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver',
                'Pancreas', 'Lung', 'Melanoma', 'Breast', 'Cervix uteri',
                'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney',
                'Bladder', 'Brain', 'Thyroid', 'NHL', 'MM', 'AML', 'Other', 'Death']

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

# train/valid/test
with open(dir_DB + 'trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)

dsplit='test'


# %%
# Functions
# ==========================================================================================
# ==========================================================================================
def round(x):
    ll = []
    for ii in range(x.shape[0]):
        hh = str(np.round(x[ii], 2))
        if len(hh)<4:
            hh = hh + '0'  
        ll.extend([hh])
    return(np.asarray(ll))

# %%
# Plot Settings
# ==========================================================================================
# ==========================================================================================
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

# %%
ci = [np.asarray(pd.read_csv(dir_out  + events[ii] + '/data/concordance_' + dsplit + '.csv', usecols=[1]))[0].tolist() for ii in range(22)]
ci_se = [np.asarray(pd.read_csv(dir_out +  events[ii] + '/data/concordance_' + dsplit + '.csv', usecols=[2]))[0].tolist() for ii in range(22)]

ci_f = [np.asarray(pd.read_csv(dir_out +  events[ii] + '/data/concordance_' + dsplit + '.csv', usecols=[1]))[1].tolist() for ii in range(22)]
ci_f_se = [np.asarray(pd.read_csv(dir_out +  events[ii] + '/data/concordance_' + dsplit + '.csv', usecols=[2]))[1].tolist() for ii in range(22)]

ci_m = [np.asarray(pd.read_csv(dir_out +  events[ii] + '/data/concordance_' + dsplit + '.csv', usecols=[1]))[2].tolist() for ii in range(22)]
ci_m_se = [np.asarray(pd.read_csv(dir_out +  events[ii] + '/data/concordance_' + dsplit + '.csv', usecols=[2]))[2].tolist() for ii in range(22)]


ci = np.asarray(ci)[:, 0]
ci_se = np.asarray(ci_se)[:, 0]

ci_f = np.asarray(ci_f)[:, 0]
ci_f_se = np.asarray(ci_f_se)[:, 0]

ci_m = np.asarray(ci_m)[:, 0]
ci_m_se = np.asarray(ci_m_se)[:, 0]


l_ci = round(ci)
l_ci_f = round(ci_f)
l_ci_m = round(ci_m)


# %%
fig, ax = plt.subplots(1, 1, figsize=(3*cm, 10*cm), dpi=600)

ax.barh(y=np.arange(0, 22, 1)+0.2, width=ci_f, height=0.4, xerr=ci_f_se, color=colormap, hatch='')
ax.barh(y=np.arange(0, 22, 1)-0.2, width=ci_m, height=0.4, xerr=ci_m_se, color=colormap, hatch='xxxxxxxxxxxxxxxxxxxx')

ax.plot(ci, np.arange(0, 22, 1), color='black', ls='', marker='|', ms=7, mew=1.2)
ax.set_xlim(0.5, 0.8)

ax.set_yticks(np.arange(0, 22, 1))
ax.set_yticklabels(np.asarray(Events))
ax.set_xlabel('Concordance')
ax.set_ylim([-0.5, 21.5])

ax0 = ax.twinx()
ax0.set_ylim(ax.get_ylim())
ax0.set_yticks(np.arange(0, 22, 1))
ax0.set_yticklabels(np.asarray(['    ' + str(l_ci[ii]) for ii in range(22)]))

ax1 = ax.twinx()
ax1.set_ylim(ax.get_ylim())
ax1.set_yticks(np.arange(0, 22, 1))
ax1.set_yticklabels(np.asarray([str(l_ci_f[ii]) + '\n' + str(l_ci_m[ii]) for ii in range(22)]), fontsize=3.4)

ax.set_xticks(np.arange(0.5, 0.8, 0.1))
ax.set_xticklabels(['0.5', '', '', '0.8'])

plt.savefig(dir_out  + 'main/figures/concordance_' + dsplit +'.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out  + 'main/figures/concordance_' + dsplit +'.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()


# %%
exit()

# %%



