'''
Pipe for extracting baseline characteristics
'''

# loading packages
import sys
import os 
import h5py
import numpy as np 
import pandas as pd

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_helper import pipe_helper

class pipe_bth(pipe_helper):
    def __init__(self, **kwds):
        """
        Doc: 
            - Pipe for extracting information for the baseline infos - bth

        """
        super().__init__(**kwds)

    def __bth__(self, fidx):
        """
        Doc: 
            

        Args:
            fidx                               hdf5 file

        Return: 
            res (np.array):                    array of baseline covariates []
                                               height, weight, systolic, diastolic, pulse,
                                               packyear, bmi, alcohol_weekly, alcohol, smoking
                                                
            date (np.date):                    date
        """
        bth = fidx['bth'][:].astype(str)
        dates = bth[:, 0].astype('datetime64[D]')
        bth = bth[:, 1:]
        bth[bth == ''] = 'nan'
        bth = bth.astype(float)
        return(dates, bth)
        
    def __bthcall__(self, fidx):
        """
        """
        dates, bth = self.__bth__(fidx=fidx)

        # heigh - weight - bmi computation
        if np.any(np.logical_and(pd.isna(bth[:, 0]), np.logical_and(~pd.isna(bth[:, 1]), ~pd.isna(bth[:, 6])))):
            bth[:, 0] = np.sqrt((bth[:, 1] / (bth[:, 6]/10000)))

        elif np.any(np.logical_and(pd.isna(bth[:, 6]), np.logical_and(~pd.isna(bth[:, 1]), ~pd.isna(bth[:, 0])))):
            bth[:, 6] = (bth[:, 1] / bth[:, 0]**2)*10000

        elif np.any(np.logical_and(pd.isna(bth[:, 1]), np.logical_and(~pd.isna(bth[:, 0]), ~pd.isna(bth[:, 6])))):
            bth[:, 1] = (bth[:, 6]/10000) * bth[:, 0]**2

        # generate final data
        res = np.round(np.concatenate((
            ((np.clip(bth[:, 0], 75, 250))/100)[:, None],                 # height 
            (((bth[:, 1])/100))[:, None],                                 # weight 
            ((np.clip(bth[:, 6], 15, 45)))[:, None],                      # bmi 
            (np.logical_or(bth[:, 2] >= 140, bth[:, 3] >= 90))[:, None],  # high blodd pressure > 140/90 
            (np.logical_and(bth[:, 2] <= 90, bth[:, 3] <= 60))[:, None],  # low blood pressure <  90/60 
            (np.logical_or(bth[:, 5] >= 5, bth[:, 9] >= 1))[:, None],     # smoking - any smoking 
            (np.logical_or(bth[:, 7] >= 20, bth[:, 8] >= 3))[:, None]     # alcohol - heavy drinker or > 20 alc_weekly
        ), axis=1), 4).astype(str)

        res[~np.logical_or(~pd.isna(bth[:, 2]), ~pd.isna(bth[:, 3])), -4] = np.nan
        res[~np.logical_or(~pd.isna(bth[:, 2]), ~pd.isna(bth[:, 3])), -3] = np.nan
        res[~np.logical_or(~pd.isna(bth[:, 5]), ~pd.isna(bth[:, 9])), -2] = np.nan
        res[~np.logical_or(~pd.isna(bth[:, 7]), ~pd.isna(bth[:, 8])), -1] = np.nan
        #res[res=='nan'] = '-9999'
        res = res.astype(float)
        # reorder 
        res = res[:, [6, 5, 3, 4, 2, 0, 1]] #alc, smoking, highBP, lowBP, bmi, heihgt, weight
        res[:, :4] = res[:, :4]*2 -1 # effect coding
        
        res[pd.isna(res)] = -9999
        res[res==0] = -9999
        
        return([dates, res])







            
            
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        