'''
'''

## Modules
#=======================================================================================================================

import sys
import os 
import tqdm
import h5py
import subprocess

import numpy as np 
import pandas as pd 
from sklearn import metrics
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt

from _custom_functions import KM, CIF

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Melanoma', 'Breast', 
                'Cervix Uteri', 'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney', 'Bladder', 'Brain', 'Thyroid', 'Non-Hodgkin Lymphoma', 'Multiple Myeloma', 'AML', 'Other', 'Death']

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
fontsize=6

colormap= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#421a01', '#6e0b3c', 
                '#9370DB', '#7A378B', '#CD6090', '#006400', '#5ebd70', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CDCB50', '#CD6600', '#FF8C69', '#8f0000', '#b3b3b3', '#454545'])


## Plots
#=======================================================================================================================

## Age-Sex Plot
#=======================================================================================================================

def Age_Sex_plot(pred, sex, age, cc, colormap=colormap, events=events, Events=Events, ROOT_DIR=ROOT_DIR):
    ll1 = []
    ll2 = []
    for ii in range(50, 75):
        idxage = np.logical_and(age/365 > ii, age/365 <=ii+5)
        if cc in [7, 8, 9, 10]:
            ll1.extend([pred[np.logical_and(~sex, idxage), 0, cc].sum()/np.logical_and(~sex, idxage).sum()])
        elif cc in [11, 12]:
            ll2.extend([pred[np.logical_and(sex, idxage), 0, cc].sum()/np.logical_and(sex, idxage).sum()])
        else:
            ll1.extend([pred[np.logical_and(~sex, idxage), 0, cc].sum()/np.logical_and(~sex, idxage).sum()])
            ll2.extend([pred[np.logical_and(sex, idxage), 0, cc].sum()/np.logical_and(sex, idxage).sum()])

    fig, ax = plt.subplots(1, 1, figsize=(4*cm, 3*cm), dpi=600)
    if cc not in [11, 12]:
        ax.plot(range(50, 75), np.asarray(ll1)*100000/5, color=colormap[cc])
    if cc not in [7, 8, 9, 10]:
        ax.plot(range(50, 75), np.asarray(ll2)*100000/5, color=colormap[cc], ls='--')
    ax.set_xlabel('Age')
    ax.set_ylabel('Rates per 100,000')
    
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/agesex.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/agesex.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()
    
    if cc in [7, 8, 9, 10]:
        ll2 = np.repeat(0, 25)
    elif cc in [11, 12]:
        ll1 = np.repeat(0, 25)

    dd = pd.DataFrame(np.concatenate((np.arange(50, 75)[:, None], 
                    np.asarray(ll1)[:, None],
                    np.asarray(ll2)[:, None]), axis=1))
    dd.columns = ['Age', 'Incidence_female', 'Incidence_male']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' +  events[cc] + '/data/agesex.csv')


## 5yr risk prediction
#=======================================================================================================================

def risk_plot_5yr(A0, cc, colormap=colormap, events=events, Events=Events, ROOT_DIR=ROOT_DIR):
    ll1=[]
    ll2=[]
    for ii in range(0, 30000, 365*5):
        ll1.extend([np.cumsum(A0[cc, 0, 0+ii:1825+ii])[-1]])
        ll2.extend([np.cumsum(A0[cc, 1, 0+ii:1825+ii])[-1]])
        
    fig, ax = plt.subplots(1, 1, figsize=(4*cm, 3*cm), dpi=600)
    ax.plot(range(0,85,5), np.asarray(ll1)*100000/5, color=colormap[cc], ls='-', label='female')
    ax.plot(range(0,85,5), np.asarray(ll2)*100000/5, color=colormap[cc], ls='--', label='male')
    ax.set_ylabel('Rates per 100,000')
    ax.set_xlabel('Age')
    ax.legend(frameon=False, fontsize=5)
    
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/yr5Risk.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/yr5Risk.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()  
    
    if cc in [7, 8, 9, 10]:
        ll2 = np.repeat(0, 17)
    elif cc in [11, 12]:
        ll1 = np.repeat(0, 17)
    
    dd = pd.DataFrame(np.concatenate((np.arange(0, 85, 5)[:, None], 
                np.asarray(ll1)[:, None],
                np.asarray(ll2)[:, None]), axis=1))
         
    dd.columns = ['Age', 'Risk_5yr_female', 'Risk_5yr_male']
    dd.to_csv(ROOT_DIR + 'projects/CancerRisk/output/' +  events[cc] + '/data/yr5Risk.csv')

## Cummulative Hazard
#=======================================================================================================================
def cumhaz_plot(A0, cc, colormap=colormap, events=events, Events=Events, ROOT_DIR=ROOT_DIR):
    fig, ax = plt.subplots(1, 1, figsize=(4*cm, 3*cm), dpi=600)

    ax.plot(np.arange(0,31400)/365, np.cumsum(A0[cc, 0, :]), color=colormap[cc], ls='-', label='female')
    ax.plot(np.arange(0,31400)/365, np.cumsum(A0[cc, 1, :]), color=colormap[cc], ls='--', label='male')
    
    ax.set_xticks([0, 25, 50, 75])
    ax.set_ylabel(r'$\Lambda_0$')
    ax.set_xlabel('Age')
    
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/cummulativeHaz.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(ROOT_DIR + 'projects/CancerRisk/output/' + events[cc] + '/figures/cummulativeHaz.png', dpi=600, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()              





