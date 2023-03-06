# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import dill as pickle
import tqdm

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats.mstats import gmean

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

# predictor
#from m1 import predictor 

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
dir_root = '/users/projects/cancer_risk/'

dtype = torch.FloatTensor 

# disease codes
disease_ref= pd.read_csv(dir_data + 'CancerRisk/disease_ref.csv', sep=';')
disease_ref = np.asarray(disease_ref.iloc[:, 2])
disease_cat = np.load(dir_data + 'CancerRisk/disease_codes_cat.npy')
dcounter= disease_cat[:, 1].astype(float)

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']


Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver',
                'Pancreas', 'Lung', 'Melanoma', 'Breast', 'Cervix uteri',
                'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney',
                'Bladder', 'Brain', 'Thyroid', 'Non Hodgkin Lymphoma', 'Multiple Myeloma', 'AML']

gene_names = np.asarray([jj+ii for jj in Events for ii in [' First Degree', ' All', ' Multiple', ' Early']])

bth_names = np.asarray(['Alcoholic', 'Smoker', 'High Blood Pressure', 'Low Blood Pressure', 'Height', 'Weight', 'Age at first Birth'])

# %%

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
sp = pd.read_csv(dir_data + 'CancerRisk/standardized_population.csv', sep=';')
sp = sp.loc[np.logical_and(sp['year']>=1995, sp['year']<2015)].reset_index(drop=True)

# agregate over years
sp = sp[['sex', 'age', 'total']].groupby(['sex', 'age']).mean().reset_index()
sp['bins'] = pd.cut(sp['age'], bins = np.arange(15, 90, step=5))

# aggregate over age bins
sp = sp[['sex', 'bins', 'total']].groupby(['sex', 'bins']).sum().reset_index()
sp.sort_values(['sex', 'bins'], inplace=True)
sp.reset_index(drop=True, inplace= True)

# %%
fig, ax = plt.subplots(1, 1, figsize=(2.5*cm, 4*cm), dpi=600)

ax.barh(y=np.arange(0, 14, 1), width=-np.asarray(sp[sp['sex']==0].loc[:, 'total']), height=0.4, color='#8044AA', hatch='')
ax.barh(y=np.arange(0, 14, 1), width=np.asarray(sp[sp['sex']==1].loc[:, 'total']), height=0.4, color='#44AA69', hatch='')


ax.set_yticks(np.arange(0, 14, 2))
ax.set_yticklabels((15 + 5*np.arange(0, 14, 2)).astype(str))

ax.set_xticks([-200000, 0,  200000])
ax.set_xticklabels(['200', '0', '200'])
ax.set_ylabel('Age')
ax.set_xlabel('Count per 1,000')

#plt.show()
plt.savefig(dir_out  + 'figures/all/population_pyramid.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out  + 'figures/all/population_pyramid.png', dpi=600, bbox_inches='tight', transparent=True)
plt.close()

sp.to_csv(dir_out  + 'figures/all/data/population_pyramid.csv', sep=';')

# %%
# save file 
with open(dir_data + 'CancerRisk/cancer_incidence.pkl', 'rb') as f:
    dicct_cancer = pickle.load(f)

# %%
for cc in range(22):
    dd = pd.DataFrame(np.asarray(dicct_cancer[Events[cc]]))
    dd.columns=['sex', 'age', 'year']
    dd['total'] = 1
    dd['bins'] = pd.cut(dd['age'], bins = np.arange(15, 90, step=5))

    # aggregate over age bins
    dd = dd[['sex', 'bins', 'total']].groupby(['sex', 'bins']).sum().reset_index()
    dd.sort_values(['sex', 'bins'], inplace=True)
    dd.reset_index(drop=True, inplace= True)
    dd['total'].fillna(0, inplace=True)
    dd['total_adj'] = (dd['total']/sp['total'])/20*100000

    fig, ax = plt.subplots(1, 1, figsize=(7*cm, 3.75*cm), dpi=600)

    if cc in [7, 8, 9, 10]:
        ax.bar(x=np.arange(14)-0.2, height=np.asarray(dd.loc[dd['sex']==0, 'total_adj']), width=0.4,hatch='', color=colormap[cc])

    elif cc in [11, 12]:
        ax.bar(x=np.arange(14)+0.2, height=np.asarray(dd.loc[dd['sex']==1, 'total_adj']), width=0.4,hatch='xxxxxxxxxxxx', color=colormap[cc])
    else:
        ax.bar(x=np.arange(14)-0.2, height=np.asarray(dd.loc[dd['sex']==0, 'total_adj']), width=0.4,hatch='', color=colormap[cc])
        ax.bar(x=np.arange(14)+0.2, height=np.asarray(dd.loc[dd['sex']==1, 'total_adj']), width=0.4,hatch='xxxxxxxxxxxx', color=colormap[cc])

    ax.set_xticks(range(0, 14, 2))
    ax.set_xticklabels(np.arange(20, 90, step=10), rotation=0)
    ax.set_xlabel('Age')
    ax.set_ylabel('Incidence per 100,000')

    #plt.show()
    plt.savefig(dir_out  + 'figures/' + events[cc] + '/agesex.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out  + 'figures/' + events[cc] + '/agesex.png', dpi=600, bbox_inches='tight', transparent=True)
    
    plt.close()

    dd.to_csv(dir_out  + 'figures/' + events[cc] + '/data/agesex.csv', sep=';')


# %%
for cc in range(22): 
    dd = pd.DataFrame(np.asarray(dicct_cancer[Events[cc]]))
    dd.columns=['sex', 'age', 'year']
    dd['total'] = 1
    
    # aggregate over years
    dd = dd[['sex', 'year', 'total']].groupby(['sex', 'year']).sum().reset_index()
    dd.sort_values(['sex', 'year'], inplace=True)
    dd.reset_index(drop=True, inplace= True)
    dd['total'].fillna(0, inplace=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(3*cm, 1.5*cm), dpi=600)
    if cc in [7, 8, 9, 10]:
        ax.plot(np.asarray(dd.loc[dd['sex']==0, 'year']), np.asarray(dd.loc[dd['sex']==0, 'total'])/1000, color=colormap[cc])
    elif cc in [11, 12]:
        ax.plot(np.asarray(dd.loc[dd['sex']==1, 'year']), np.asarray(dd.loc[dd['sex']==1, 'total'])/1000, ls='--',dashes=(1.5, 0.75), color=colormap[cc])
    else:
        ax.plot(np.asarray(dd.loc[dd['sex']==0, 'year']), np.asarray(dd.loc[dd['sex']==0, 'total'])/1000, color=colormap[cc])
        ax.plot(np.asarray(dd.loc[dd['sex']==1, 'year']), np.asarray(dd.loc[dd['sex']==1, 'total'])/1000, ls='--',dashes=(1.5, 0.75), color=colormap[cc])

    ax.set_xticks([1995, 2015])
    ax.set_xticklabels(['1995', '2015'], rotation=0)
    ax.set_xlabel('Years')
    ax.set_ylabel('Incidence per 1,000')

    plt.savefig(dir_out  + 'figures/' + events[cc] + '/incidence.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out  + 'figures/' + events[cc] + '/incidence.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()

    dd.to_csv(dir_out  + 'figures/' + events[cc] + '/data/incidence.csv', sep=';')


# %%
exit()


