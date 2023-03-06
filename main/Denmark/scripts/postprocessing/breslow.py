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
# Baseline Hazard 
# Custom funtions
# =======================================================================================================================
class A0_fun():
    def __init__(self, tt, basehaz):
        self.tt = tt
        self.basehaz = basehaz
        
    def __call__(self, ii):
        
        if np.sum(ii > self.tt) == 0:
            return(0)
        elif np.sum(ii > self.tt) <= len(self.tt):
            return(self.basehaz[np.sum(ii > self.tt)-1][0])
        else:
            return(self.basehaz[-1][0])
A0 = [dill.load(open(dir_out + 'model/' + events[event_idx] + '/breslow.pkl', 'rb')) for event_idx in range(22)]
A0_net = np.asarray([[np.sum([A0[ii]['female'](jj) for ii in range(22)]), np.sum([A0[ii]['male'](jj) for ii in range(22)])] for jj in (range(32000))])
S0_net = np.exp(-np.cumsum(A0_net, axis=0))

# %%
for event_idx in range(22):

    A0 = dill.load(open(dir_out + 'model/' + events[event_idx] + '/breslow.pkl', 'rb'))  

    fig, ax = plt.subplots(1, 1, figsize=(4.5*cm, 3.5*cm), dpi=600)
    #ax.plot(range(365*80), np.cumsum([A0['female'](ii) for ii in range(365*80)]), color=colormap[event_idx], ls=(0, (3, 1, 1, 1, 1, 1)))
    #ax.plot(range(365*80), np.cumsum([A0['male'](ii) for ii in range(365*80)]), color=colormap[event_idx], ls=(0, (3, 10, 1, 10, 1, 10)))

    ax.plot(range(365*80), -np.log(1-np.cumsum([A0['female'](ii)*S0_net[ii, 0] for ii in range(365*80)])), color=colormap[event_idx])
    ax.plot(range(365*80), -np.log(1-np.cumsum([A0['male'](ii)*S0_net[ii, 1] for ii in range(365*80)])), ls='--',dashes=(1.5, 0.75), color=colormap[event_idx])

    
    ax.set_xticks(np.arange(0, 365*82, 365*20))
    ax.set_xticklabels(np.arange(0, 90, 20))
    
    ax.set_ylabel('Cumulative Hazard')
    ax.set_xlabel('Age')

    plt.savefig(dir_out  + 'figures/' + events[event_idx] + '/breslow.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out  + 'figures/' + events[event_idx] + '/breslow.png', dpi=600, bbox_inches='tight', transparent=True)
    
    plt.show()
    plt.close()

# %%
exit()


