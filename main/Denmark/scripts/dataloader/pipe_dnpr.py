'''
Pipe for extracting information from dnpr 

'''
# loading packages
import sys
import os 
import h5py

import numpy as np 

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_helper import pipe_helper


class pipe_dnpr(pipe_helper):
    def __init__(self,  **kwds):
        """
        Doc: 
            - Pipe for extracting information from dnpr 

        """
        super().__init__(**kwds)
        
        
    def __dnpr__(self, fidx):
        """
        Doc: 
            Retrieves DNPR information contained in the HDF5 for individual idx

        Args:
            fidx:      hdf5 file
           

        Return: 
            date (np.datetime)(None, 1): Date of birth
            diagnosis (np.str)(None, 2): ICD10 diagnosis code + diagnosis type
        """
        
        dnpr = fidx['DNPR'][:]
        date = dnpr[:, 0].astype('datetime64')
        diagnosis = dnpr[:, (1, 2)].astype('str')
        return(date, diagnosis)
        
        
    def __dnprcall__(self, fidx):
        """
        Doc: 
            Minor adjustments for DNPR data

        Args:
            fidx                         hdf5 file

        Return: 
            dnpr_date:                          Dates for diagnosis 
            dnpr_diagnosis:                     Diagnosis codes

        """
        
        dnpr_date, dnpr_diagnosis = self.__dnpr__(fidx=fidx)
        dnpr_idx = np.repeat(True, dnpr_date.shape[0])
        
        
        dnpr_bth = np.zeros((0, 7)).astype(float) # #alc, smoking, highBP, lowBP, heihgt, weight, bmi
        dates_dnpr_bth = np.asarray([]).astype('datetime64[D]')

        ###### Extract Additional Info:
        # Birth indicator
        idx_1 = self.__in__(dnpr_diagnosis[:, 0], ['DZ37', 'DZ370', 'DZ371', 'DZ372', 'DZ373' 'DZ374', 'DZ375', 'DZ376', 'DZ377' 'DZ378', 'DZ379',
'DZ39', 'DZ390', 'DZ391', 'DZ392', 'DZ393' 'DZ394', 'DZ395', 'DZ396', 'DZ397' 'DZ398', 'DZ399',
        'DO80', 'DO800', 'DO801', 'DO802', 'DO803' 'DO804', 'DO805', 'DO806', 'DO807' 'DO808', 'DO809',
'DO81', 'DO810', 'DO811', 'DO812', 'DO813' 'DO814', 'DO815', 'DO816', 'DO817' 'DO818', 'DO819',
'DO82', 'DO820', 'DO821', 'DO822', 'DO823' 'DO824', 'DO825', 'DO826', 'DO827' 'DO828', 'DO829',
'DO83', 'DO830', 'DO831', 'DO832', 'DO833' 'DO834', 'DO835', 'DO836', 'DO837' 'DO838', 'DO839',
'DO84', 'DO840', 'DO841', 'DO842', 'DO843' 'DO844', 'DO845', 'DO846', 'DO847' 'DO848', 'DO849'])
        if np.sum(idx_1) >=1:
            dnpr_child = np.min(dnpr_date[idx_1])[None]
        else: 
            dnpr_child = np.zeros((0,)).astype('datetime64[D]')
            
        #### 4th level codes    
        # extract alcohol information for bth
        helpvar_alc = self.__in__(dnpr_diagnosis[:, 0], ['DG621', 'DG6312', 'DG721', 'DI426', 'DK292', 'DK860', 'DZ714', 'DZ502', 'DZ721'])
        if np.sum(helpvar_alc) > 0:
            try:
                helpvar_alc_d = np.min(dnpr_date[helpvar_alc])[None]
            except:
                helpvar_alc_d = np.asarray([]).astype('datetime64[D]')
                
        else:
            helpvar_alc_d = np.asarray([]).astype('datetime64[D]')
            
        # extract smoking information for bth
        helpvar_smok = self.__in__(dnpr_diagnosis[:, 0], ['DZ508', 'DZ716', 'DZ720'])
        if np.sum(helpvar_smok) > 0:  
            helpvar_smok_d = np.min(dnpr_date[helpvar_smok])[None]
        else:
            helpvar_smok_d = np.asarray([]).astype('datetime64[D]')
            

        ##### adjustment
        # primary/secondary diagnosis
        dnpr_idx = dnpr_idx * self.__in__(dnpr_diagnosis[:, 1], ['A', 'A,B', 'A,B,G', 'A,C', 'A,C,B', 'A,G', 'A,H', 'A,H,B', 'A,H,B,M', 'A,H,C', 'A,H,G', 'A,H,M', 'A,M', 'B', 'B,G', 'C,B', 'H,B', 'H,B,G'])
        
        # unmatched icd8 codes
        dnpr_idx = dnpr_idx * np.asarray([var[:1] == 'D' for var in dnpr_diagnosis[:, 0]]) 
        
        # remove chapters - in accordance with UKB 1712 - No external factors.
        dnpr_idx = dnpr_idx * np.asarray([var[:2] not in ['DT', 'DS', 'DU', 'DV', 'DX', 'DY', 'DW', 'DC'] for var in dnpr_diagnosis[:, 0]])
        # remove unwanted obs
        dnpr_date = dnpr_date[dnpr_idx]
        dnpr_diagnosis = dnpr_diagnosis[dnpr_idx, 0]
        
        # transform code to international icd10
        dnpr_diagnosis = np.asarray([var[1:4] for var in dnpr_diagnosis]) 

        # keep unique codes with minimum date
        dnpr_date = np.asarray([np.min(dnpr_date[dnpr_diagnosis == dd]) for dd in np.unique(dnpr_diagnosis)])
        dnpr_diagnosis = np.unique(dnpr_diagnosis)
        
        dnpr_date = dnpr_date.astype('datetime64[D]')
        
        # bth additional information
        alc_hh = self.__in__(dnpr_diagnosis, ['F10', 'K70'])
        if np.sum(alc_hh) + np.sum(helpvar_alc) > 0: 
            if dnpr_diagnosis.shape[0] > 0:
                alc_d = np.min(np.concatenate((dnpr_date[alc_hh], helpvar_alc_d)))[None]
            else:
                alc_d = helpvar_alc_d
            dnpr_bth = np.concatenate((dnpr_bth, np.asarray([[1, -9999, -9999, -9999, -9999, -9999, -9999]])), axis=0)
            dates_dnpr_bth = np.concatenate((dates_dnpr_bth, alc_d))

        smok_hh = self.__in__(dnpr_diagnosis, ['F17'])
        if np.sum(smok_hh) + np.sum(helpvar_smok) > 0:  
            if dnpr_diagnosis.shape[0] > 0:
                smok_d = np.min(np.concatenate((dnpr_date[smok_hh], helpvar_smok_d)))[None]
            else:
                smok_d = helpvar_smok_d
            dnpr_bth = np.concatenate((dnpr_bth, np.asarray([[-9999, 1, -9999, -9999, -9999, -9999, -9999]])), axis=0)
            dates_dnpr_bth = np.concatenate((dates_dnpr_bth, smok_d))    

        HBP_hh = self.__in__(dnpr_diagnosis, ['I10', 'I11', 'I12', 'I13', 'I15'])
        if np.sum(HBP_hh) > 0:  
            HBP_d = np.min(dnpr_date[HBP_hh])[None]
            dnpr_bth = np.concatenate((dnpr_bth, np.asarray([[-9999, -9999, 1, -9999, -9999, -9999, -9999]])), axis=0)
            dates_dnpr_bth = np.concatenate((dates_dnpr_bth, HBP_d))  

        LBP_hh = self.__in__(dnpr_diagnosis, ['I95'])
        if np.sum(LBP_hh) > 0:  
            LBP_d = np.min(dnpr_date[LBP_hh])[None]
            dnpr_bth = np.concatenate((dnpr_bth, np.asarray([[-9999, -9999, -9999, 1, -9999, -9999, -9999]])), axis=0)
            dates_dnpr_bth = np.concatenate((dates_dnpr_bth, LBP_d))  

        return(dnpr_date, dnpr_diagnosis, dates_dnpr_bth, dnpr_bth, dnpr_child)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        