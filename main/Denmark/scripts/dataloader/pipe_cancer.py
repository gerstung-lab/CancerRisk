'''
Pipe for extracting and making a decision on cancers
'''

# loading packages
import sys
import os 
import h5py
import numpy as np 

from itertools import chain

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_helper import pipe_helper

class pipe_cancer(pipe_helper):
    def __init__(self, **kwds):
        """
        Doc: 
            - Pipe for extracting information for cancer

        """
        self.malignant_codes =  np.concatenate([['C0' + str(_) for _ in range(10)],
        ['C' + str(_) for _ in range(10,100)], 
        ['Z08', 'Z85']])

        self.uncertain_cancer = np.asarray(['D' + str(_) for _ in range(37, 49)])
        
        self.cancers = [['C15'], ['C16'], ['C18', 'C19', 'C20', 'C21'],
           ['C22'], ['C25'], ['C33', 'C34'], ['C43'],
           ['C50'], ['C53'], ['C54'], ['C56'],
           ['C61'], ['C62'], ['C64'], ['C67'],
           ['C71'], ['C73'], ['C82', 'C83', 'C84', 'C85', 'C86'],
           ['C90'], ['C920', 'C924', 'C925', 'C926', 'C928', 'C930', 'C940', 'C942']]
        
        self.cancers.append( list(set(np.concatenate([['C0' + str(_) for _ in range(10)],
        ['C' + str(_) for _ in range(10,100)]]).tolist()).difference(set(list(chain.from_iterable(self.cancers)))))
        )

        self.cancer_names = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other']

        super().__init__(**kwds)

    def __cancer__(self, fidx):
        """
        Doc: 
            retrieves Cancer information contained in the HDF5 for individual idx

        Args:
            fidx                        hdf5 file

        Return: 
            cset (np.str)(None, 3):      'date', 'icd10', 'count', 'tumour_count', 'morph', 'idx_dnpr', 'idx_cr', 'idx_dr'

        """
 
        cset_full = fidx['cancer']['set_full'][:].astype(str)
            
        return(cset_full)
        
    def __cancercall__(self, fidx):
        """
        Doc: 
            Calling cancer decision for idx in file.

        Args:
            idx (str):                         PID for individual
            file (int):                        File number

        Return: 
            decision (list):                   list of indicators corresponding to cancer
            date (np.date):                    first incidence of a relevant cancer - defined by cancer_codes 
        """
            
        # cancer set
        cancer_set = self.__cancer__(fidx=fidx)
        # standardice cancer entry to 4levles at max
        cancer_set[:, 1] = np.asarray([kk[:4] for kk in cancer_set[:, 1]])
        idx_cancer = self.__in__(cancer_set[:, 1], self.malignant_codes, reduce_x=True)
        
        # if relevant cancer present
        if np.sum(idx_cancer) > 0:
            
            # subset to relevant codes 
            primary = ((cancer_set[idx_cancer, 0].astype('datetime64[D]') - np.min(cancer_set[idx_cancer, 0].astype('datetime64[D]'))).astype(int) < 31*3)
            date = np.min(cancer_set[idx_cancer, 0].astype('datetime64[D]'))
            
            # possibly adjust by uncertain cancer diagnosis date (if less then 1 year apart)
            idx_uncertain = self.__in__(cancer_set[:, 1], self.uncertain_cancer, reduce_x=True)
            if np.sum(idx_uncertain) > 0:
                date_uncertain = np.min((cancer_set[idx_uncertain, 0]).astype('datetime64[D]'))
                if (date - date_uncertain).astype(int) <= 365:  
                    date = np.minimum(date, date_uncertain)

            cancer_set = cancer_set[idx_cancer, :]
            cancer_set = cancer_set[primary, :]
            cancer_set[:, 0] = date

            # remove if morphology indicates metastatic cancer
            cancer_set = cancer_set[~np.asarray([cancer_set[kk, 4][-1] in ['6'] for kk in range(cancer_set.shape[0])])]
            
            # remove - (only CR - classified as benign or in situ)
            if cancer_set.shape[0] > 0:
                idx_remv = (((cancer_set[:, -3] == '0') * (cancer_set[:, -2] == '1') * (cancer_set[:, -1] == '0') * np.asarray([cancer_set[kk, 4][-1] in ['2', '0'] for kk in range(cancer_set.shape[0])])))
                cancer_set = cancer_set[~idx_remv, :]
            
            # remove - (only CR - classified as benign or in situ)
            if cancer_set.shape[0] > 0:
                idx_remv = (((cancer_set[:, -3] == '0') * (cancer_set[:, -2] == '1') * (cancer_set[:, -1] == '0') * np.asarray([cancer_set[kk, 4][-1] in ['9', '1'] for kk in range(cancer_set.shape[0])])))
                if not np.all(idx_remv):
                    cancer_set = cancer_set[~idx_remv, :]
                

            decision = np.max(np.asarray([[np.logical_or(cc[:-1] in kk, cc in kk) for kk in self.cancers] for cc in cancer_set[:, 1]]), axis=0).astype(int)
            # preferable selct only 20 major cancers - ignoring Bag of Cancers
            if np.sum(decision[:20]) >= 1: 
                decision[-1] = 0
            
        else:
            decision, date, cancer_set = None, None, None
        return(date, decision, cancer_set)




            
            
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        