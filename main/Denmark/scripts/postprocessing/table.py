# %% [markdown]
# # Summary Characteristics Table:

# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import pickle
import torch 
import tqdm

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from torch.utils.data import DataLoader
from multiprocessing import Pool

# seeding
np.random.seed(7)
torch.random.manual_seed(8)

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'


with open(dir_data + 'CancerRisk/samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)

disease_codes = np.load(dir_data + 'CancerRisk/disease_codes.npy', allow_pickle=True)


events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

cancers = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

mode='test2'

# %%
# Custom Functions
# ==========================================================================================
# ==========================================================================================
def IQR(x, idx=None):
    if np.any(idx==None):
        a, b, c = np.round(np.quantile(x, [0.05, 0.5, 0.95]), 4)
    else:
        a, b, c = np.round(np.quantile(x[idx], [0.05, 0.5, 0.95]), 4)
        
    return(['[ ' + str(a) + ' - ' + str(b) + ' - ' + str(c) + ' ]'])
    

def __in__(x, y, reduce_x=False):
    """
    Doc: 
        Compares whether the entries from on list (main) appear in the other list (secondary)

    Args:
        x (list):                   Entires in main list
        y (list):                   Entires in secondary list
        reduce_x(bool):             Whether to cut of part of x ( -> 3rd level icd10 code)

    Return: 
        helpvar (np.bool)(1,):      Array with logical indicators  
    """
    x = np.asarray(x)
    y = np.asarray(y)
    helpvar = []
    if reduce_x:
        for ii in x:
            if len(ii) == 4:
                helpvar.append(ii[:-1] in y)
            else:
                helpvar.append(ii in y)   
    else:
        for ii in x: 
            helpvar.append(ii in y)
    return(np.asarray(helpvar))

def prep_dnpr(dnpr_unadj):
        dnpr_date, dnpr_diagnosis = dnpr_unadj
        dnpr_idx = np.repeat(True, dnpr_date.shape[0])
        
        ##### adjustment
        # primary/secondary diagnosis
        dnpr_idx = dnpr_idx * __in__(dnpr_diagnosis[:, 1], ['A', 'A,B', 'A,B,G', 'A,C', 'A,C,B', 'A,G', 'A,H', 'A,H,B', 'A,H,B,M', 'A,H,C', 'A,H,G', 'A,H,M', 'A,M', 'B', 'B,G', 'C,B', 'H,B', 'H,B,G'])
        
        # unmatched icd8 codes
        dnpr_idx = dnpr_idx * np.asarray([var[:1] == 'D' for var in dnpr_diagnosis[:, 0]]) 
        
        # remove chapters - in accordance with UKB 1712 - No external factors.
        dnpr_idx = dnpr_idx * np.asarray([var[:2] not in ['DT', 'DS', 'DU', 'DV', 'DX', 'DY', 'DW', 'DC', 'DZ'] for var in dnpr_diagnosis[:, 0]])
        # remove unwanted obs
        dnpr_date = dnpr_date[dnpr_idx]
        dnpr_diagnosis = dnpr_diagnosis[dnpr_idx, 0]
        
        # transform code to international icd10
        dnpr_diagnosis = np.asarray([var[1:4] for var in dnpr_diagnosis]) 
        return(dnpr_date, dnpr_diagnosis)
    
def eval_dnpr(dnpr_unadj, EOO, birthdate):
    # DNPR Infos
    a, b = prep_dnpr(dnpr_unadj)
    diagnosis = b.shape[0]
    diagnosisU = np.unique(b).shape[0]
    lifeyears = ((np.minimum(EOO, np.datetime64('2015-01-01')) - np.maximum(birthdate, np.datetime64('1979-01-01'))).astype(int)/365)[0]
    hospital_visits = np.unique(a).shape[0]
    return(np.asarray([diagnosis, diagnosisU, lifeyears, hospital_visits]))      

# %%
# Pipe 
# ==========================================================================================
# ==========================================================================================
sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_cancer import pipe_cancer
from pipe_dnpr import pipe_dnpr
from pipe_genealogy import pipe_genealogy
from pipe_bth import pipe_bth

PIPE_cancer = pipe_cancer()
PIPE_dnpr = pipe_dnpr()
PIPE_genealogy = pipe_genealogy()
PIPE_bth = pipe_bth()

from dataloader import Data_Pipeline
PIPE = Data_Pipeline(event_idx=None, sex_specific=np.asarray([0, 1]))

# %%
# train/valid/test
with open(dir_data + 'DB/DB/raw/trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)

# %%
mode = 'train'
if mode == 'train':
    # proportions
    with open(dir_data + 'CancerRisk/proportions.pickle', 'rb') as handle:
        proportions = pickle.load(handle)
elif mode == 'valid':
    # proportions
    with open(dir_data + 'CancerRisk/proportions_valid.pickle', 'rb') as handle:
        proportions = pickle.load(handle)
elif mode == 'test':
    # proportions
    with open(dir_data + 'CancerRisk/proportions_test.pickle', 'rb') as handle:
        proportions = pickle.load(handle)

if mode in np.asarray(['train', 'valid', 'test']):
    cancer_counts = []
    for cc in range(7):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + str(int(proportions[3][cc])))
    for cc in range(7, 11):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + '0')
    for cc in range(7, 9):
        cancer_counts.append('0' + ' - ' + str(int(proportions[3][cc])))
    for cc in range(11, 20):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + str(int(proportions[3][cc-2])))

    sampling_proportion = []
    for cc in range(7):
        sampling_proportion.append([proportions[0]+proportions[2], None, proportions[1][cc]+proportions[3][cc], None])
    for cc in range(7, 11):
            sampling_proportion.append([proportions[0], None, proportions[1][cc], None])
    for cc in range(7, 9):
            sampling_proportion.append([proportions[2], None, proportions[3][cc], None])
    for cc in range(11, 20):
        sampling_proportion.append([proportions[0]+proportions[2], None, proportions[1][cc]+proportions[3][cc-2], None])


# %%
if mode == 'train':
    # proportions
    with open(dir_data + 'CancerRisk/proportions.pickle', 'rb') as handle:
        proportions = pickle.load(handle)
elif mode == 'valid':
    # proportions
    with open(dir_data + 'CancerRisk/proportions_valid.pickle', 'rb') as handle:
        proportions = pickle.load(handle)
elif mode == 'test':
    # proportions
    with open(dir_data + 'CancerRisk/proportions_test.pickle', 'rb') as handle:
        proportions = pickle.load(handle)

if mode in np.asarray(['train', 'valid', 'test']):
    cancer_counts = []
    for cc in range(7):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + str(int(proportions[3][cc])))
    for cc in range(7, 11):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + '0')
    for cc in range(7, 9):
        cancer_counts.append('0' + ' - ' + str(int(proportions[3][cc])))
    for cc in range(11, 20):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + str(int(proportions[3][cc-2])))

    sampling_proportion = []
    for cc in range(7):
        sampling_proportion.append([proportions[0]+proportions[2], None, proportions[1][cc]+proportions[3][cc], None])
    for cc in range(7, 11):
            sampling_proportion.append([proportions[0], None, proportions[1][cc], None])
    for cc in range(7, 9):
            sampling_proportion.append([proportions[2], None, proportions[3][cc], None])
    for cc in range(11, 20):
        sampling_proportion.append([proportions[0]+proportions[2], None, proportions[1][cc]+proportions[3][cc-2], None])


    ll_dnpr = []
    ll_gene = []
    ll_bth = []
    ll_base = []

    n_m = 0
    n_f = 0 
    n = 0


    for ii in tqdm.tqdm(tvt_split[mode]): #tvt_split['train']
        ii1 = ii//100
        ii2 = (ii - ii1*100)//10
        file = root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, ii)
        with h5py.File(file, 'r') as f:
            idx_list = list(f.keys())
            for jj in tqdm.tqdm(samples[ii]):
                n += 1
                fidx = f[idx_list[jj]]
                dnpr_unadj = PIPE_dnpr.__dnpr__(fidx=fidx)
                birthdate, sex, status, EOO, SOO, idxcancer = PIPE_cancer.__maincall__(fidx=fidx)
                if sex==0:
                    n_f +=1
                else:
                    n_m +=1
                dates_cancer, decision, cancer_set = PIPE_cancer.__cancercall__(fidx=fidx)
                dates_dnpr, dnpr, dates_dnpr_bth, dnpr_bth, dnpr_child = PIPE_dnpr.__dnprcall__(fidx=fidx)
                dates_genealogy, genealogy, genealogy_kids, genealogy_base = PIPE_genealogy.__genealogycall__(fidx=fidx, SOO=SOO, EOO=EOO, birthdate=birthdate)
                G = PIPE_genealogy.__genealogy__(fidx=fidx)
                dates_bth, bth  = PIPE_bth.__bth__(fidx=fidx)
                out_bth = PIPE_bth.__bthcall__(fidx=fidx)
                hh = out_bth[1]
                hh[hh==-9999] = np.nan
                out = PIPE.__infer__(fidx=fidx)

                genealogy_birthdates, genealogy_degree, genealogy_EOO, genealogy_sex, genealogy_set, genealogy_children, genealogy_kinsman = PIPE_genealogy.__genealogy__(fidx=fidx)


                ###### Time Adjustments
                # death date 
                early_EOO = EOO <= np.datetime64('2014-12-31')

                # Time - adjustments - observation limit 2015 / age limit 75 / remove last year of obs prior to cancer
                if dates_cancer != None: # remove last year before cancer 
                    EOO = np.minimum(EOO, np.datetime64('2014-12-31'))
                    EOO = np.minimum(EOO, birthdate + 365*86)
                    EOO = np.minimum(EOO, dates_cancer)
                    EOO = EOO - 365 
                else: # remove possible last year - (as this information would not be availbe if a cancer would have been present)
                    EOO = np.minimum(EOO, np.datetime64('2014-12-31'))
                    EOO = np.minimum(EOO, birthdate + 365*86)
                    EOO = EOO - 365 

                #DNPR
                if dnpr.shape[0] == 0:
                    dates_dnpr = np.zeros((0,)).astype('datetime64[D]')
                    dnpr = np.zeros((0,)).astype(str)
                else:
                    idx_r = dates_dnpr <= EOO
                    dates_dnpr = dates_dnpr[idx_r]
                    dnpr = dnpr[idx_r]
                    dates_dnpr = np.maximum(dates_dnpr, SOO)

                #Genealogy 
                idx_r = dates_genealogy <= EOO
                dates_genealogy = dates_genealogy[idx_r]
                genealogy = genealogy[idx_r]
                dates_genealogy = np.maximum(dates_genealogy, SOO)

                # BTH 
                idx_r = dates_bth <= EOO
                dates_bth = dates_bth[idx_r]
                bth = bth[idx_r]
                hh = hh[idx_r]
                dates_bth = np.maximum(dates_bth, SOO)

                helpvar_dnpr = eval_dnpr(dnpr_unadj, EOO, birthdate)
                ll_dnpr.extend([helpvar_dnpr])

                helpvar_gene = [genealogy_base[0]*10, genealogy_base[1]*100, genealogy_base[2]*10, genealogy_base[3]*100]
                helpvar_gene.extend([G[0].shape[0]])
                helpvar_gene.extend(np.max(out[2], axis=0).tolist())
                ll_gene.extend([helpvar_gene])

                helpvar_bth = []
                if np.sum(~pd.isna(dates_bth))> 0:
                    bth_visits = np.unique(dates_bth).shape[0]
                    bth_lifeyears = ((EOO - np.min(dates_bth)).astype(float)/365)[0]
                else: 
                    bth_visits = 0 
                    bth_lifeyears = 0
                helpvar_bth.extend([bth_visits, bth_lifeyears])


                helpvar_bth.extend(np.sum(hh == hh, axis=0).tolist())    
                #helpvar_bth.extend(np.nanmean(bth, axis=0).tolist())
                try:
                    helpvar_bth.extend(np.nanmax(hh, axis=0).tolist())
                except:
                    helpvar_bth.extend(np.repeat(np.nan, 7).tolist())


                ll_bth.extend([helpvar_bth])
                ll_base.extend(np.concatenate((sex[:, None], np.asarray(out[3][-1, -1])[None, None]), axis=1).tolist())

    ll_dnpr = np.asarray(ll_dnpr)
    ll_base = np.asarray(ll_base)
    ll_bth = np.asarray(ll_bth)
    ll_gene = np.asarray(ll_gene)

    idx_dnpr_noshowups = ll_dnpr[:, 3] !=0
    idx_nofamily = ll_gene[:, 0] != 0
    idx_nofamily_1st = ll_gene[:, 2] != 0
    idx_family = ll_gene[:, 4] != 0


    res = pd.DataFrame()
    res['Individuals'] = [n]
    res['Female - Male'] = str(n_f) + ' - ' +str(n_m) 
    res['Never Showups'] = (ll_dnpr[:, 3] ==0).sum().astype(float).astype(str)
    res['Never Showups - %'] = np.round((ll_dnpr[:, 3] ==0).sum().astype(float)/ll_dnpr[:, 3].shape[0], 4)
    res['Hospital Visits'] = ll_dnpr[:, 3].sum().astype(float).astype(str)
    res['Hospital Visits - IQR'] = IQR(ll_dnpr[:, 3], idx=idx_dnpr_noshowups)
    res['Diagnoses'] = ll_dnpr[:, 0].sum().astype(float).astype(str)
    res['Diagnoses - IQR'] = IQR(ll_dnpr[:, 0], idx=idx_dnpr_noshowups)
    res['Unique Diagnoses'] = ll_dnpr[:, 1].sum().astype(float).astype(str)
    res['Unique Diagnoses - IQR'] = IQR(ll_dnpr[:, 1], idx=idx_dnpr_noshowups)
    res['Lifeyears'] = np.floor(ll_dnpr[:, 2].sum().astype(float)).astype(str)
    res['Lifeyears - IQR'] = IQR((ll_dnpr[:, 2]), idx=idx_dnpr_noshowups)
    res['Mothers'] = str((ll_base[:, 1] > 0).sum()) + ' - ' + str(np.round((ll_base[:, 1] > 0).sum()/(ll_base[:, 0]==0).sum(), 4))
    res['Age at first Birth - IQR'] = IQR((ll_base[:, 1][ll_base[:, 1] > 0] * 100), idx=None)

    cancers = 0 
    for cc in range(21):
        cancers += sampling_proportion[cc][2]

    res['cancer'] = str(int(cancers)) + ' (' +  str(np.round(cancers/sampling_proportion[0][0], 5))   +')'
    for cc in range(len(events)):
        res['C -' + events[cc]] = cancer_counts[cc]

    res['Family Info'] = str(np.sum(idx_family)) + ' - ' + str(np.sum(idx_family)/idx_family.shape[0])
    res['F/M Info'] = str(np.sum(idx_nofamily)) + ' - ' + str(np.sum(idx_nofamily)/idx_nofamily.shape[0])
    res['Family Members - IQR'] = IQR(ll_gene[idx_nofamily, 0])
    res['Family Age - IQR'] = IQR(ll_gene[idx_nofamily, 1])
    res['1stD Family Members - IQR'] = IQR(ll_gene[idx_nofamily_1st, 2])
    res['1stD Family Age - IQR'] = IQR(ll_gene[idx_nofamily_1st, 3])

    res['Genealogy'] = 'First Degree Relative - All Relatives - Multiple Cases - Early (<50yrs) Case'
    for cc in range(20):
        res['G -' + events[cc]] = str(int(np.sum(ll_gene[:, 5+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 5+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ') - ' +str(int(np.sum(ll_gene[:, 6+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 6+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ') - ' +str(int(np.sum(ll_gene[:, 7+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 7+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ') - ' + str(int(np.sum(ll_gene[:, 8+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 8+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ')'

    res['BTH Info'] = str(np.sum(ll_bth[:, 0] > 0)) + ' (' + str(np.round(np.sum(ll_bth[:, 0] > 0)/ll_bth.shape[0], 3)) + ')'
    res['BTH Visits'] = str(np.sum(ll_bth[:, 0]))
    res['BTH Visits - IQR'] = IQR(ll_bth[:, 0].astype(float), idx=ll_bth[:, 0]>0)
    res['BTH Lifeyears'] = np.floor(ll_bth[:, 1].sum().astype(float)).astype(str)
    res['BTH Lifeyears - IQR'] = IQR((ll_bth[:, 1]), idx=ll_bth[:, 1]>0)
    res['BTH Alcoholic/Non-Alcoholic'] = str(np.sum(ll_bth[:, -7]==1)) + ' / ' + str(np.sum(ll_bth[:, -7]==-1))
    res['BTH Ever Smoker/Non-Smoker'] =str(np.sum(ll_bth[:, -6]==1)) + ' / ' + str(np.sum(ll_bth[:, -6]==-1))
    res['BTH High Blood Pressure/Non-HBP'] =str(np.sum(ll_bth[:, -5]==1)) + ' / ' + str(np.sum(ll_bth[:, -5]==-1))
    res['BTH Low Blood Pressure/Non-LBP'] =str(np.sum(ll_bth[:, -4]==1)) + ' / ' + str(np.sum(ll_bth[:, -4]==-1))
    res['Height - IQR'] =IQR(ll_bth[:, -2], idx=~pd.isna(ll_bth[:, -2]))
    res['Weight - IQR'] =IQR(ll_bth[:, -1], idx=~pd.isna(ll_bth[:, -1]))

    res.T.to_csv(dir_tab + 'tab1_' +  mode +'.csv', sep=';')



# %%
def eval_dnpr(dnpr_unadj, EOO, birthdate):
    # DNPR Infos
    a, b = prep_dnpr(dnpr_unadj)
    diagnosis = b.shape[0]
    diagnosisU = np.unique(b).shape[0]
    lifeyears = ((EOO - np.maximum(birthdate, np.datetime64('1979-01-01'))).astype(int)/365)[0]
    hospital_visits = np.unique(a).shape[0]
    return(np.asarray([diagnosis, diagnosisU, lifeyears, hospital_visits]))    

if mode == 'test2':
    # proportions
    with open(dir_data + 'CancerRisk/proportions_test2.pickle', 'rb') as handle:
        proportions = pickle.load(handle)

    cancer_counts = []
    for cc in range(7):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + str(int(proportions[3][cc])))
    for cc in range(7, 11):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + '0')
    for cc in range(7, 9):
        cancer_counts.append('0' + ' - ' + str(int(proportions[3][cc])))
    for cc in range(11, 20):
        cancer_counts.append(str(int(proportions[1][cc])) + ' - ' + str(int(proportions[3][cc-2])))
    
    sampling_proportion = []
    for cc in range(7):
        sampling_proportion.append([proportions[0]+proportions[2], None, proportions[1][cc]+proportions[3][cc], None])
    for cc in range(7, 11):
            sampling_proportion.append([proportions[0], None, proportions[1][cc], None])
    for cc in range(7, 9):
            sampling_proportion.append([proportions[2], None, proportions[3][cc], None])
    for cc in range(11, 20):
        sampling_proportion.append([proportions[0]+proportions[2], None, proportions[1][cc]+proportions[3][cc-2], None])
        
    ll_dnpr = []
    ll_gene = []
    ll_bth = []
    ll_base = []
    n_m = 0
    n_f = 0 
    n = 0
    for ii in tqdm.tqdm(range(1000)): #tvt_split['train']
        ii1 = ii//100
        ii2 = (ii - ii1*100)//10
        file = dir_DB + 'f_%i/f_%i/_%i' %(ii1, ii2, ii)
        with h5py.File(file, 'r') as f:
            idx_list = list(f.keys())
            for jj in (samples[ii]):
                fidx = f[idx_list[jj]]
                dnpr_unadj = PIPE_dnpr.__dnpr__(fidx=fidx)
                birthdate, sex, status, EOO, SOO, idxcancer = PIPE_cancer.__maincall__(fidx=fidx)
                dates_cancer, decision, cancer_set = PIPE_cancer.__cancercall__(fidx=fidx)
                
                
                if dates_cancer:
                    if dates_cancer <= np.datetime64('2015-01-01'):
                        continue
                        
                if EOO <= np.datetime64('2015-01-01'):
                    continue
                    
                if np.logical_or((np.datetime64('2015-01-01') - birthdate).astype(float) < 16*365, (np.datetime64('2015-01-01') - birthdate).astype(float) > 75*365):
                    continue
                    
                n += 1
                if sex==0:
                    n_f +=1
                else:
                    n_m +=1
                
                dates_dnpr, dnpr, dates_dnpr_bth, dnpr_bth, dnpr_child = PIPE_dnpr.__dnprcall__(fidx=fidx)
                dates_genealogy, genealogy, genealogy_kids, genealogy_base = PIPE_genealogy.__genealogycall__(fidx=fidx, SOO=SOO, EOO=EOO, birthdate=birthdate)
                G = PIPE_genealogy.__genealogy__(fidx=fidx)
                dates_bth, bth  = PIPE_bth.__bth__(fidx=fidx)
                out_bth = PIPE_bth.__bthcall__(fidx=fidx)
                hh = out_bth[1]
                hh[hh==-9999] = np.nan
                out = PIPE.__infer__(fidx=fidx)

                genealogy_birthdates, genealogy_degree, genealogy_EOO, genealogy_sex, genealogy_set, genealogy_children, genealogy_kinsman = PIPE_genealogy.__genealogy__(fidx=fidx)
                
                ###### Time Adjustments
                # death date 
                early_EOO = EOO < np.datetime64('2018-04-09')

                # Time - adjustments - observation limit 2015 / remove last year of obs prior to cancer
                if dates_cancer != None: 
                    EOO = np.minimum(EOO, np.datetime64('2014-12-31'))
                    EOO = np.minimum(EOO, dates_cancer)
                    EOO = EOO - 365 
                else: # remove possible last year - (as this information would not be availbe if a cancer would have been present)
                    EOO = np.minimum(EOO, np.datetime64('2014-12-31'))
                    EOO = EOO - 365 

                #DNPR
                if dnpr.shape[0] == 0:
                    dates_dnpr = np.zeros((0,)).astype('datetime64[D]')
                    dnpr = np.zeros((0,)).astype(str)
                else:
                    idx_r = dates_dnpr <= EOO
                    dates_dnpr = dates_dnpr[idx_r]
                    dnpr = dnpr[idx_r]
                    dates_dnpr = np.maximum(dates_dnpr, SOO)

                #Genealogy 
                idx_r = dates_genealogy <= EOO
                dates_genealogy = dates_genealogy[idx_r]
                genealogy = genealogy[idx_r]
                dates_genealogy = np.maximum(dates_genealogy, SOO)

                # BTH 
                idx_r = dates_bth <= EOO
                dates_bth = dates_bth[idx_r]
                bth = bth[idx_r]
                hh = hh[idx_r]
                dates_bth = np.maximum(dates_bth, SOO)

                helpvar_dnpr = eval_dnpr(dnpr_unadj, EOO, birthdate)
                ll_dnpr.extend([helpvar_dnpr])

                helpvar_gene = [genealogy_base[0]*10, genealogy_base[1]*100, genealogy_base[2]*10, genealogy_base[3]*100]
                helpvar_gene.extend([G[0].shape[0]])
                helpvar_gene.extend(np.max(out[2], axis=0).tolist())
                ll_gene.extend([helpvar_gene])

                helpvar_bth = []
                if np.sum(~pd.isna(dates_bth))> 0:
                    bth_visits = np.unique(dates_bth).shape[0]
                    bth_lifeyears = ((EOO - np.min(dates_bth)).astype(float)/365)[0]
                else: 
                    bth_visits = 0 
                    bth_lifeyears = 0
                helpvar_bth.extend([bth_visits, bth_lifeyears])


                helpvar_bth.extend(np.sum(hh == hh, axis=0).tolist())    
                try:
                    helpvar_bth.extend(np.nanmax(hh, axis=0).tolist())
                except:
                    helpvar_bth.extend(np.repeat(np.nan, 7).tolist())
                    

                ll_bth.extend([helpvar_bth])
                ll_base.extend(np.concatenate((sex[:, None], np.asarray(out[3][-1, -1])[None, None]), axis=1).tolist())

    ll_dnpr = np.asarray(ll_dnpr)
    ll_base = np.asarray(ll_base)
    ll_bth = np.asarray(ll_bth)
    ll_gene = np.asarray(ll_gene)

    idx_dnpr_noshowups = ll_dnpr[:, 3] !=0
    idx_nofamily = ll_gene[:, 0] != 0
    idx_nofamily_1st = ll_gene[:, 2] != 0
    idx_family = ll_gene[:, 4] != 0

    res = pd.DataFrame()
    res['Individuals'] = [n]
    res['Female - Male'] = str(n_f) + ' - ' +str(n_m) 
    res['Never Showups'] = (ll_dnpr[:, 3] ==0).sum().astype(float).astype(str)
    res['Never Showups - %'] = np.round((ll_dnpr[:, 3] ==0).sum().astype(float)/ll_dnpr[:, 3].shape[0], 4)
    res['Hospital Visits'] = ll_dnpr[:, 3].sum().astype(float).astype(str)
    res['Hospital Visits - IQR'] = IQR(ll_dnpr[:, 3], idx=idx_dnpr_noshowups)
    res['Diagnoses'] = ll_dnpr[:, 0].sum().astype(float).astype(str)
    res['Diagnoses - IQR'] = IQR(ll_dnpr[:, 0], idx=idx_dnpr_noshowups)
    res['Unique Diagnoses'] = ll_dnpr[:, 1].sum().astype(float).astype(str)
    res['Unique Diagnoses - IQR'] = IQR(ll_dnpr[:, 1], idx=idx_dnpr_noshowups)
    res['Lifeyears'] = np.floor(ll_dnpr[:, 2].sum().astype(float)).astype(str)
    res['Lifeyears - IQR'] = IQR((ll_dnpr[:, 2]), idx=idx_dnpr_noshowups)
    res['Mothers'] = str((ll_base[:, 1] > 0).sum()) + ' - ' + str(np.round((ll_base[:, 1] > 0).sum()/(ll_base[:, 0]==0).sum(), 4))
    res['Age at first Birth - IQR'] = IQR((ll_base[:, 1][ll_base[:, 1] > 0] * 100), idx=None)

    cancers = 0 
    for cc in range(21):
        cancers += sampling_proportion[cc][2]

    res['cancer'] = str(int(cancers)) + ' (' +  str(np.round(cancers/sampling_proportion[0][0], 5))   +')'
    for cc in range(len(events)):
        res['C -' + events[cc]] = cancer_counts[cc]
        
    res['Family Info'] = str(np.sum(idx_family)) + ' - ' + str(np.sum(idx_family)/idx_family.shape[0])
    res['F/M Info'] = str(np.sum(idx_nofamily)) + ' - ' + str(np.sum(idx_nofamily)/idx_nofamily.shape[0])
    res['Family Members - IQR'] = IQR(ll_gene[idx_nofamily, 0])
    res['Family Age - IQR'] = IQR(ll_gene[idx_nofamily, 1])
    res['1stD Family Members - IQR'] = IQR(ll_gene[idx_nofamily_1st, 2])
    res['1stD Family Age - IQR'] = IQR(ll_gene[idx_nofamily_1st, 3])

    res['Genealogy'] = 'First Degree Relative - All Relatives - Multiple Cases - Early (<50yrs) Case'
    for cc in range(20):
        res['G -' + events[cc]] = str(int(np.sum(ll_gene[:, 5+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 5+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ') - ' +str(int(np.sum(ll_gene[:, 6+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 6+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ') - ' +str(int(np.sum(ll_gene[:, 7+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 7+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ') - ' + str(int(np.sum(ll_gene[:, 8+(4*cc)]==1))) + ' (' + str(np.round(np.sum(ll_gene[:, 8+(4*cc)]==1) / np.sum(idx_nofamily), 4)) + ')'

    res['BTH Info'] = str(np.sum(ll_bth[:, 0] > 0)) + ' (' + str(np.round(np.sum(ll_bth[:, 0] > 0)/ll_bth.shape[0], 3)) + ')'
    res['BTH Visits'] = str(np.sum(ll_bth[:, 0]))
    res['BTH Visits - IQR'] = IQR(ll_bth[:, 0].astype(float), idx=ll_bth[:, 0]>0)
    res['BTH Lifeyears'] = np.floor(ll_bth[:, 1].sum().astype(float)).astype(str)
    res['BTH Lifeyears - IQR'] = IQR((ll_bth[:, 1]), idx=ll_bth[:, 1]>0)
    res['BTH Alcoholic/Non-Alcoholic'] = str(np.sum(ll_bth[:, -7]==1)) + ' / ' + str(np.sum(ll_bth[:, -7]==-1))
    res['BTH Ever Smoker/Non-Smoker'] =str(np.sum(ll_bth[:, -6]==1)) + ' / ' + str(np.sum(ll_bth[:, -6]==-1))
    res['BTH High Blood Pressure/Non-HBP'] =str(np.sum(ll_bth[:, -5]==1)) + ' / ' + str(np.sum(ll_bth[:, -5]==-1))
    res['BTH Low Blood Pressure/Non-LBP'] =str(np.sum(ll_bth[:, -4]==1)) + ' / ' + str(np.sum(ll_bth[:, -4]==-1))
    res['Height - IQR'] =IQR(ll_bth[:, -2], idx=~pd.isna(ll_bth[:, -2]))
    res['Weight - IQR'] =IQR(ll_bth[:, -1], idx=~pd.isna(ll_bth[:, -1]))

    res.T.to_csv('/users/projects/cancer_risk/main/output/main/tables/tab1_test2.csv', sep=';')

    

# %%
exit()

# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N table
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=80gb
#PBS -l walltime=150:00:00

cd $PBS_O_WORDIR
module load anaconda3/2021.05
source conda activate

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/postprocessing/table.ipynb --output /users/projects/cancer_risk/main/scripts/postprocessing/table

/services/tools/anaconda3/2021.05/bin/python3.8 /users/projects/cancer_risk/main/scripts/postprocessing/table.py $VAR1
' >> run.sh

qsub run.sh


# %%



