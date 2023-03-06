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
def adj(x, xlower, xupper, xlower2, xupper2, names):
    num=25

    idxsig = np.sign(xlower) == np.sign(xupper)
    x = x[idxsig]
    xlower = xlower[idxsig]
    xupper = xupper[idxsig]
    xlower2 = xlower2[idxsig]
    xupper2 = xupper2[idxsig]
    names = names[idxsig]
    
    xsig = np.argsort(np.abs(x))
    x = x[xsig]
    xlower = xlower[xsig]
    xupper = xupper[xsig]
    xlower2 = xlower2[xsig]
    xupper2 = xupper2[xsig]
    names = names[xsig]
    
    x = x[-num:]
    xlower = xlower[-num:]
    xupper = xupper[-num:]
    xlower2 = xlower2[-num:]
    xupper2 = xupper2[-num:]
    names = names[-num:]
    
    xsig = np.sign(xlower2) == np.sign(xupper2)
    
    x = np.exp(x)
    xlower = x - np.exp(xlower)
    xupper = np.exp(xupper) - x
    xlower2 = x - np.exp(xlower2)
    xupper2 = np.exp(xupper2) - x
    
    idxsort = np.argsort(x)
    x = x[idxsort]
    xlower  = xlower[idxsort]
    xupper = xupper[idxsort]
    xlower2  = xlower2[idxsort]
    xupper2 = xupper2[idxsort]
    names = names[idxsort]
    xsig = xsig[idxsort]
    
    
    return(x, xlower, xupper, names, xsig, xlower2, xupper2)

# %%
tt = [pickle.load(open(dir_root + 'main/model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]
for cc in tqdm.tqdm(range(22)):
    print(events[cc])
    theta_dnpr_lower  = tt[cc]['guide'].quantiles([0.05])['theta_dnpr'][0].detach().numpy()[0, :]
    theta_gene_lower = tt[cc]['guide'].quantiles([0.05])['theta_gene'][0].detach().numpy()[0, :]
    theta_bth_lower = tt[cc]['guide'].quantiles([0.05])['theta_bth'][0].detach().numpy()[0, :]
    
    theta_dnpr_lower2  = tt[cc]['guide'].quantiles([0.005])['theta_dnpr'][0].detach().numpy()[0, :]
    theta_gene_lower2 = tt[cc]['guide'].quantiles([0.005])['theta_gene'][0].detach().numpy()[0, :]
    theta_bth_lower2 = tt[cc]['guide'].quantiles([0.005])['theta_bth'][0].detach().numpy()[0, :]

    theta_dnpr = tt[cc]['guide'].quantiles([0.5])['theta_dnpr'][0].detach().numpy()[0, :]
    theta_gene = tt[cc]['guide'].quantiles([0.5])['theta_gene'][0].detach().numpy()[0, :]
    theta_bth = tt[cc]['guide'].quantiles([0.5])['theta_bth'][0].detach().numpy()[0, :]

    theta_dnpr_upper = tt[cc]['guide'].quantiles([0.95])['theta_dnpr'][0].detach().numpy()[0, :]
    theta_gene_upper = tt[cc]['guide'].quantiles([0.95])['theta_gene'][0].detach().numpy()[0, :]
    theta_bth_upper = tt[cc]['guide'].quantiles([0.95])['theta_bth'][0].detach().numpy()[0, :]
    
    theta_dnpr_upper2 = tt[cc]['guide'].quantiles([0.995])['theta_dnpr'][0].detach().numpy()[0, :]
    theta_gene_upper2 = tt[cc]['guide'].quantiles([0.995])['theta_gene'][0].detach().numpy()[0, :]
    theta_bth_upper2 = tt[cc]['guide'].quantiles([0.995])['theta_bth'][0].detach().numpy()[0, :]
    
    # rescaling estimates
    # height/weight per 10 cm/kg
    theta_bth_lower[-3:] = theta_bth_lower[-3:]/10
    theta_bth_lower2[-3:] = theta_bth_lower2[-3:]/10
    theta_bth[-3:] = theta_bth[-3:]/10
    theta_bth_upper[-3:] = theta_bth_upper[-3:]/10
    theta_bth_upper2[-3:] = theta_bth_upper2[-3:]/10
    
    x, xlower, xupper, names, xsig, xlower2, xupper2 = adj(x=np.concatenate((theta_dnpr, theta_gene, theta_bth)), xlower=np.concatenate((theta_dnpr_lower, theta_gene_lower, theta_bth_lower)), xupper=np.concatenate((theta_dnpr_upper, theta_gene_upper, theta_bth_upper)), xlower2=np.concatenate((theta_dnpr_lower2, theta_gene_lower2, theta_bth_lower2)), xupper2=np.concatenate((theta_dnpr_upper2, theta_gene_upper2, theta_bth_upper2)), names=np.concatenate((disease_ref, gene_names, bth_names)))
    
    sigcolor = np.asarray([('#363636', colormap[cc])[xsig[ii]] for ii in range(x.shape[0])])
    
    fig, ax = plt.subplots(1, 1, figsize=(5*cm, x.shape[0]*0.72*cm), dpi=600)#18cm
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.0)

    ax.errorbar(y=range(x.shape[0]), x=x, xerr=(xlower, xupper), color='#363636', linestyle='', marker='', linewidth=2, capthick=1.5, capsize=2)
    for ii in range( x.shape[0]):
        ax.plot(x[ii], ii, color=sigcolor[ii], marker='.', markersize=5)

    ax.axvline(1, linewidth=0.5, color='black', ls=':')
    ax.set_yticks(range(x.shape[0]))
    ax.set_yticklabels(names)
    ax0 = ax.twinx()
    ax0.set_ylim(ax.get_ylim())
    ax0.set_yticks(range(x.shape[0]))
    ax0.set_yticklabels(['(' + (x[ii]-xlower[ii]).astype(str)[:4] + ' < '  + (x[ii]).astype(str)[:4] + ' > '  + (x[ii]+xupper[ii]).astype(str)[:4] + ')' for ii in range(len(names))])
    
    ax.set_xlabel('Hazard Ratio')

    plt.savefig(dir_out + events[cc] + '/figures/forest.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/forest.png', dpi=600, bbox_inches='tight', transparent=True)
    
    plt.show()
    plt.close()   
    
    dd = pd.DataFrame(np.concatenate(((x-xlower2)[:, None], (x-xlower)[:, None], x[:, None], (x+xupper)[:, None], (x+xupper2)[:, None], names[:, None]), axis=1))
    dd.columns = ['q0005', 'q005', 'median', 'q095', 'q0995', 'names']
    dd.to_csv(dir_out + events[cc] + '/data/forest.csv', sep=';')
    

# %%
exit()

# %%


# %%



