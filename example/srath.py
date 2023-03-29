
# Modules
# =======================================================================================================================
import pickle
import sys
import torch 
import pyro 

import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt


events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

for cc in range(22):
    print(cc, events[cc])




disease_codes = np.load('//Users/alexwjung/Desktop/CancerRisk/model/disease_codes.npy', allow_pickle=True)
gene_names = np.asarray([jj+ii for jj in events[:20] for ii in [' First Degree', ' All', ' Multiple', ' Early']])
bth_names = np.asarray(['Alcoholic', 'Smoker', 'High Blood Pressure', 'Low Blood Pressure', 'Height', 'Weight', 'Age at first Birth'])

sys.path.append('//Users/alexwjung/Desktop/CancerRisk/model/')
from m1 import predictor




names = np.concatenate((disease_codes[:, 1], gene_names, bth_names))



#=======================================================================================================================
tt = [pickle.load(open('//Users/alexwjung/Desktop/model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]



th = []
colname = ['covariate']

for cc in (range(len(tt))):
    pyro.clear_param_store()
    with torch.no_grad(): 
        #tt = pickle.load(open(dir_out + 'm1/model/' + events[cc] + '/param.pkl', 'rb'))
        mm = tt[cc]['model']        
        gg = tt[cc]['guide']

        theta_dnpr = gg.quantiles([0.5])['theta_dnpr'][0].detach().numpy()[0, :]
        theta_gene = gg.quantiles([0.5])['theta_gene'][0].detach().numpy()[0, :]
        theta_bth = gg.quantiles([0.5])['theta_bth'][0].detach().numpy()[0, :]
        th.extend([np.concatenate((theta_dnpr, theta_gene, theta_bth)).tolist()])
        colname.extend([events[cc] + '_theta'])
        
        theta_dnpr = gg.quantiles([0.025])['theta_dnpr'][0].detach().numpy()[0, :]
        theta_gene = gg.quantiles([0.025])['theta_gene'][0].detach().numpy()[0, :]
        theta_bth = gg.quantiles([0.025])['theta_bth'][0].detach().numpy()[0, :]
        th.extend([np.concatenate((theta_dnpr, theta_gene, theta_bth)).tolist()])
        colname.extend([events[cc] + '_theta_0.025'])
        
        theta_dnpr = gg.quantiles([0.975])['theta_dnpr'][0].detach().numpy()[0, :]
        theta_gene = gg.quantiles([0.975])['theta_gene'][0].detach().numpy()[0, :]
        theta_bth = gg.quantiles([0.975])['theta_bth'][0].detach().numpy()[0, :]
        th.extend([np.concatenate((theta_dnpr, theta_gene, theta_bth)).tolist()])
        colname.extend([events[cc] + '_theta_0.975'])
        

th = np.asarray(th).T


th.shape

dd = pd.DataFrame(np.concatenate((names[:, None], th), axis=1))


dd.columns = colname

dd
dd.to_csv(