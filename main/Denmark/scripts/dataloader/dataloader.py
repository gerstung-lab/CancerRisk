'''
# Dataloader - Fit
'''
# loading packages
import sys
import os 
import h5py
import pickle
import numpy as np 
import tqdm

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')

from pipe_dnpr import pipe_dnpr
from pipe_cancer import pipe_cancer
from pipe_genealogy import pipe_genealogy
from pipe_bth import pipe_bth

dir_data = '/users/projects/cancer_risk/data/'

with open(dir_data + 'CancerRisk/samples.pickle', 'rb') as handle:
    samples = pickle.load(handle)
    
disease_codes = np.load(dir_data + 'CancerRisk/disease_codes.npy', allow_pickle=True)

locations = pickle.load(open(dir_data + 'CancerRisk/locations.pickle', 'rb'))

root_dir = dir_data + 'DB/DB/raw/'

class Data_Pipeline(pipe_dnpr, pipe_cancer, pipe_genealogy, pipe_bth):
    def __init__(self, root_dir=root_dir, samples=samples, disease_codes=disease_codes, time_dependent=False, inference=True, locations=locations, event_idx=None, sex_specific=None, **kwds):
        self.samples = samples
        self.disease_codes = disease_codes
        self.root_dir = root_dir
        self.inference = inference
        self.locations = locations
        self.event_idx = event_idx
        self.sex_specific = sex_specific

        super().__init__(**kwds)
        
    def __infer__(self, fidx):
        ###### Load Data
        birthdate, sex, status, EOO, SOO, idxcancer = self.__maincall__(fidx=fidx)
        if sex in self.sex_specific:
            dates_cancer, decision, cancer_set = self.__cancercall__(fidx=fidx)
            dates_dnpr, dnpr, dates_dnpr_bth, dnpr_bth, dnpr_child = self.__dnprcall__(fidx=fidx)
            dates_genealogy, genealogy, genealogy_kids, genealogy_base = self.__genealogycall__(fidx=fidx, SOO=SOO, EOO=EOO, birthdate=birthdate)
            dates_bth, bth = self.__bthcall__(fidx=fidx)

            # combine info from bth and dnpr
            dates_bth = np.concatenate((dates_bth, dates_dnpr_bth))
            bth = np.concatenate((bth, dnpr_bth))

            if self.inference:
                meta = []
            else:
                meta = [birthdate, sex, status, EOO, dates_cancer, decision, cancer_set]

            ###### Time Adjustments
            # death date 
            early_EOO = EOO <= np.datetime64('2014-12-31')
            early_d = (EOO-birthdate).astype(int)

            # Time - adjustments - observation limit 2015 / age limit 75 / remove last year of obs prior to cancer
            if dates_cancer != None: # remove last year before cancer 
                EOO = np.minimum(EOO, np.datetime64('2014-12-31'))
                EOO = np.minimum(EOO, birthdate + 365*86)
                EOO = np.minimum(EOO, dates_cancer)
                EOO = EOO - 365 # remove last 365 days
            else: # remove possible last year - (as this information would not be availbe if a cancer would have been present)
                EOO = np.minimum(EOO, np.datetime64('2014-12-31'))
                EOO = np.minimum(EOO, birthdate + 365*86)
                EOO = EOO - 365 # remove last 365 days

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
            dates_bth = np.maximum(dates_bth, SOO)

            ###### Combine Data
            dates = np.concatenate((birthdate, SOO, dates_dnpr, dates_genealogy, dates_bth, EOO),axis=0)
            dates_idx = np.argsort(dates)
            dates = dates[dates_idx]

            # dnpr expansion
            n_exp1 = 2
            n_exp2 = dates_genealogy.shape[0] + dates_bth.shape[0] + 1
            dnpr = np.concatenate((np.repeat('', n_exp1), dnpr, np.repeat('', n_exp2)))
            dnpr = dnpr[dates_idx]

            # genealogy expansion 
            n_exp1 = 2 + dates_dnpr.shape[0] 
            n_exp2 = dates_bth.shape[0] + 1
            genealogy = np.concatenate((np.zeros((n_exp1, 20, 4)), genealogy, np.zeros((n_exp2, 20, 4))))
            genealogy = genealogy[dates_idx, :, :]

            # bth expansion
            n_exp1 = 2 + dates_dnpr.shape[0] + dates_genealogy.shape[0]
            n_exp2 = 1
            bth = np.concatenate(((np.ones((n_exp1, 7))*-9999), bth, (np.ones((n_exp2, 7))*-9999)))
            bth = bth[dates_idx]

            # dnpr collapse
            dnpr_dummies = (dnpr[:, None] == self.disease_codes).astype(float)
            dnpr_dummies = np.asarray([np.sum(dnpr_dummies[jj == dates], axis=0) for jj in np.unique(dates)])

            # genealogy collapse
            genealogy = np.concatenate([genealogy[:, ii, :] for ii in range(20)], axis=1)
            genealogy = np.asarray([np.sum(genealogy[jj == dates], axis=0) for jj in np.unique(dates)])

            # bth collapse
            bth = np.asarray([np.max(bth[jj == dates], axis=0) for jj in np.unique(dates)])

            dates = np.sort(np.unique(dates))

            # dnpr
            dnpr_dummies = np.minimum(np.cumsum(dnpr_dummies, axis=0), 1)

            #genealogy
            genealogy = np.minimum(np.cumsum(genealogy, axis=0), 1)

            # bth
            if bth[:1, -2]==-9999:
                bth[:1, -2]=np.asarray([165, 177])[sex]/100
            if bth[:1, -1]==-9999:
                bth[:1, -1]= np.asarray([69, 82])[sex]/100   
            bth[:1, -2] -= 170/100
            bth[:1, -1] -= 75/100
                
            for ii in range(1, bth.shape[0]):
                bth[ii, bth[ii, :] == -9999] = bth[ii-1, bth[ii, :] == -9999]
                bth[ii, :4] = np.maximum(bth[ii, :4], bth[ii-1, :4])
                
            bth[bth==-9999] = 0
            
            
            # add BMI info to obesity codet
            dnpr_dummies[:, 307] = np.maximum((bth[:, -3]>30).astype(float), dnpr_dummies[:, 307])
            bth = np.concatenate((bth[:, :4], bth[:, -2:]), axis=1)
                
            ###### Format Data
            # long format time
            diff = (dates[1:] - dates[:-1]).astype(int)
            timeline = np.concatenate((np.cumsum(np.concatenate((np.zeros((1,)), diff)))[:-1][:, None], np.cumsum(np.concatenate((np.zeros((1,)), diff)))[1:][:, None]), axis=1)

            idx_start = np.sum(timeline[:, 0] < np.maximum(0, (SOO - birthdate).astype(float)))
            timeline = timeline[idx_start:, :] 
            dnpr_dummies = dnpr_dummies[idx_start:-1, :]
            genealogy = genealogy[idx_start:-1, :]
            bth = bth[idx_start:-1, :]
            
            # baseline
            age_first_birth = np.concatenate((dnpr_child, genealogy_kids))
            if age_first_birth.shape[0] == 0:
                age_first_birth = np.zeros((1,))
            else:
                age_first_birth = np.clip((np.min(age_first_birth) - birthdate).astype(float)/365/100, 0.1, 0.45)
            age_first_birth = age_first_birth * (1-sex)

            age_first_birth = np.repeat(age_first_birth, timeline.shape[0]).astype(float)[:, None]
            age_first_birth = age_first_birth * ((age_first_birth * 100 * 365) <= timeline[:, 0, None]).astype(float)
            
            bth = np.concatenate((bth, age_first_birth), axis=1)

            genealogy = genealogy * 2 - 1
            if genealogy_base[1] <= 0.15:
                genealogy = np.maximum(genealogy, 0)

            # cancer decision
            cancer = np.zeros((timeline.shape[0], 22))
            if dates_cancer != None:
                if dates_cancer <= np.datetime64('2014-12-31'):
                    if (dates_cancer - birthdate).astype(int) <= 365*86:
                        cancer[-1:, :21] = decision.astype(float)

            # add non-cancer death 
            if early_EOO:
                if np.sum(cancer[-1:, :21]) == 0:
                    if early_d <= 365*86:
                        cancer[-1, -1] = (status==90).astype(float)
            
            # sex stratification
            timeline = timeline + (sex*100000)
            
            # combine with cancer
            if self.event_idx!=None:
                time = np.concatenate((timeline, cancer[:, self.event_idx, None]), axis=1)
            else:
                time = np.concatenate((timeline, cancer), axis=1)

            res = [time, dnpr_dummies, genealogy, bth, meta]  
        else:
            res = [np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([]), []]
        return(res)     

    def __getitem__(self, ii):
        res = [[], [], [], [], []]
        timeline = []
        dnpr_dummies = []
        genealogy = []
        bth = []
        meta = []

        if self.inference:
            ii1 = ii//100
            ii2 = (ii - ii1*100)//10
            file = self.root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, ii)

            with h5py.File(file, 'r') as f:
                idx_list = list(f.keys())

                a, b = np.unique(np.concatenate((np.random.choice(self.locations[self.event_idx][ii], np.minimum(len(self.locations[self.event_idx][ii]), 4), replace=True), np.random.choice(self.samples[ii], 620, replace=False))),return_index=True)
                a = a[np.argsort(b)].astype(int)

                kk = 0
                ss = 0
                while ss < 90:
                    res = self.__infer__(fidx=f[idx_list[a[kk]]])
                    timeline.extend(res[0].tolist())
                    dnpr_dummies.extend(res[1].tolist())
                    genealogy.extend(res[2].tolist())
                    bth.extend(res[3].tolist())
                    kk += 1
                    if res[0].shape[0]>0:
                        ss += 1
            return([timeline, dnpr_dummies, genealogy, bth, meta])

        else:
            ii1 = ii[0]//100
            ii2 = (ii[0] - ii1*100)//10
            file = self.root_dir + 'f_%i/f_%i/_%i' %(ii1, ii2, ii[0])

            with h5py.File(file, 'r') as f:
                idx_list = list(f.keys())
                res = self.__infer__(fidx=f[idx_list[ii[1]]])
                timeline.extend(res[0].tolist())
                dnpr_dummies.extend(res[1].tolist())
                genealogy.extend(res[2].tolist())
                bth.extend(res[3].tolist())
                meta.extend(res[4])
            
            return([timeline, dnpr_dummies, genealogy, bth, meta])
 