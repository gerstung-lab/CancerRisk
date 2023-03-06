'''
Pipe for extractin the genealogy information for every family member
'''

# loading packages
import sys
import os 
import h5py
import numpy as np  

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_helper import pipe_helper
from itertools import chain

nordcan = np.load('/users/projects/cancer_risk/data/CancerRisk/nordcan.npy', allow_pickle=True)

class pipe_genealogy(pipe_helper):
    def __init__(self, nordcan=nordcan, **kwds):
        """
        Doc: 
             -

        Args:
             cancers:                         cancers considered for genealogy indicators
             nordcan:                         age-sex rate to compute potential

        Return: 
            -
        """
        
        self.genealogy_cancers = [['C15'], ['C16'], ['C18', 'C19', 'C20', 'C21'], ['C22'], ['C25'], ['C33', 'C34'], ['C43'], ['C50'], ['C53'], ['C54'], ['C56', 'C57'], ['C61'], ['C62'], ['C64'], ['C65', 'C66', 'C67', 'C68'], ['C70', 'C71', 'C72'], ['C73'], ['C82', 'C83', 'C84', 'C85', 'C86'], ['C90'], ['C91', 'C92', 'C93', 'C94', 'C95']]
        
        self.nordcan = nordcan
        
        super().__init__(**kwds)
        
    def _extract_genealogy(self, fidx, kinsman_path, kinsman_identifyer):
        """
        Doc: 
            Helper function to retrieve relevant genealogy information for relative

        Args:
            fidx (hdf5):                  HDF5 file pointer
            kinsman_path (str):           HDF5 folder for relative ex. 'father/siblings/...'
            kinsman_identifyer (str):     Adding indicator for relation 

        Return: 
            res (np.str)(None, 6):        Dates cancer degree age sex indicator
            birthdate (np.int)(1,):       Date of birth
        """
        
        gidx = fidx['genealogy'][kinsman_path]
        birthdate = gidx.attrs['birthyear'].astype('datetime64')
        EOO = gidx.attrs['EOO'].astype('datetime64')
        cancer = gidx['cancer'][:, 0].astype(str)
        morph = gidx['cancer'][:, -1].astype(str)
        dates = gidx['cancer'][:, 1].astype('datetime64')
        degree = gidx.attrs['degree'].astype(float)
        sex = gidx.attrs['sex']
        age = (dates - birthdate).astype(int)
        kinsman_identifyer_ = np.repeat(kinsman_identifyer, cancer.shape[0])
        kinsman_identifyer = np.repeat(kinsman_identifyer, birthdate.shape[0])
        res = np.concatenate((dates[:, None].astype(str), cancer[:, None].astype(str), morph[:, None].astype(str), np.repeat(degree, cancer.shape[0])[:, None].astype(str), age[:, None].astype(str), np.repeat(sex, cancer.shape[0])[:, None].astype(str), kinsman_identifyer_[:, None].astype(str)),axis=1)
          
        return([res, birthdate, degree, EOO, sex, np.asarray([kinsman_identifyer])]) 

    def __genealogy_extract__(self, fidx):
                """
                Doc: 
                    Retrieves genealogy - cancer information contained in the HDF5 for individual idx

                Args:
                    fidx                      hdf5 file

                Return: 
                    Dict.

                    **self._extract_genealogy** return for every relative - varying size relatives are returned as lists
                    basic terminology:
                    f - father
                    m - mother
                    c - children 
                    s - siblings 

                    - stack indicators together to go through family tree: ex grand aunt mother - mother side - mms
                    - returns: 
                    dict['m'] [Cancer_History, Birthdate, Degree, EOO, Sex]
                """
                genealogy = {}
                
                # mother 
                genealogy['m'] = self._extract_genealogy(fidx, 'mother',  kinsman_identifyer='m')

                # father 
                genealogy['f'] = self._extract_genealogy(fidx, 'father',  kinsman_identifyer='f')

                # grandparents
                genealogy['mm'] = self._extract_genealogy(fidx, 'mother/mother', kinsman_identifyer='mm')
                genealogy['mf'] =  self._extract_genealogy(fidx, 'mother/father', kinsman_identifyer='mf')
                genealogy['fm'] =  self._extract_genealogy(fidx, 'father/mother', kinsman_identifyer='fm')
                genealogy['ff'] =  self._extract_genealogy(fidx, 'father/father', kinsman_identifyer='ff')

                #children
                c_chist = []
                c_birthdate = []
                c_degree = []
                c_EOO = []
                c_sex = []
                c_kinsman = [] 
                for kid in np.arange(fidx['genealogy']['children'].__len__()):
                    helpvar_1, helpvar_2, helpvar_3, helpvar_4, helpvar_5, helpvar_6 = self._extract_genealogy(fidx, 'children/_' + str(kid),  kinsman_identifyer='c' + str(kid))
                    c_chist.append(helpvar_1)
                    c_birthdate.append(helpvar_2)
                    c_degree.append(helpvar_3)
                    c_EOO.append(helpvar_4)
                    c_sex.append(helpvar_5)
                    c_kinsman.append(helpvar_6)
                genealogy['c']=[c_chist, c_birthdate, c_degree, c_EOO, c_sex, c_kinsman]

                # siblings - half siblings - nephews
                s_chist = []
                s_birthdate = []
                s_degree = []
                s_EOO = []
                s_sex = []
                s_kinsman = []
                sc_chist = []
                sc_birthdate = []
                sc_degree = []
                sc_EOO = []
                sc_sex = []
                sc_kinsman = []
                for sib in range(len(fidx['genealogy']['siblings'])):
                    helpvar_1, helpvar_2, helpvar_3, helpvar_4, helpvar_5, helpvar_6 = self._extract_genealogy(fidx, 'siblings/_' + str(sib),  kinsman_identifyer='s' + str(sib))
                    s_chist.append(helpvar_1)
                    s_birthdate.append(helpvar_2)
                    s_degree.append(helpvar_3)
                    s_EOO.append(helpvar_4)
                    s_sex.append(helpvar_5)
                    s_kinsman.append(helpvar_6)
                    for kid in range(len(fidx['genealogy']['siblings/_' + str(sib) + '/children'])):
                        helpvar_1, helpvar_2, helpvar_3, helpvar_4, helpvar_5, helpvar_6 = self._extract_genealogy(fidx, 'siblings/_' + str(sib) +'/children/_' + str(kid),  kinsman_identifyer='s' + str(sib) + 'c' + str(kid))
                        sc_chist.append(helpvar_1)
                        sc_birthdate.append(helpvar_2)
                        sc_degree.append(helpvar_3)
                        sc_EOO.append(helpvar_4)
                        sc_sex.append(helpvar_5)
                        sc_kinsman.append(helpvar_6)
                genealogy['s']=[s_chist, s_birthdate, s_degree, s_EOO, s_sex, s_kinsman]
                genealogy['sc']=[sc_chist, sc_birthdate, sc_degree, sc_EOO, sc_sex, sc_kinsman]

                # aunt-uncle - cousin - mothers side
                ms_chist = []
                ms_birthdate = []
                ms_degree = []
                ms_EOO = []
                ms_sex = []
                ms_kinsman = []

                for sib in range(len(fidx['genealogy']['mother/siblings'])):
                    helpvar_1, helpvar_2, helpvar_3, helpvar_4, helpvar_5, helpvar_6 = self._extract_genealogy(fidx, 'mother/siblings/_' + str(sib), kinsman_identifyer='ms' + str(sib))
                    ms_chist.append(helpvar_1)
                    ms_birthdate.append(helpvar_2)
                    ms_degree.append(helpvar_3)
                    ms_EOO.append(helpvar_4)
                    ms_sex.append(helpvar_5)
                    ms_kinsman.append(helpvar_6)

                genealogy['ms']=[ms_chist, ms_birthdate, ms_degree, ms_EOO, ms_sex, ms_kinsman]

                # aunt-uncle - cousin - fathers side
                fs_chist = []
                fs_birthdate = []
                fs_degree = []
                fs_EOO = []
                fs_sex = []
                fs_kinsman = []

                for sib in range(len(fidx['genealogy']['father/siblings'])):
                    helpvar_1, helpvar_2, helpvar_3, helpvar_4, helpvar_5, helpvar_6 = self._extract_genealogy(fidx, 'father/siblings/_' + str(sib),  kinsman_identifyer='fs' + str(sib))
                    fs_chist.append(helpvar_1)
                    fs_birthdate.append(helpvar_2)
                    fs_degree.append(helpvar_3)
                    fs_EOO.append(helpvar_4)
                    fs_sex.append(helpvar_5)
                    fs_kinsman.append(helpvar_6)

                genealogy['fs']=[fs_chist, fs_birthdate, fs_degree, fs_EOO, fs_sex, fs_kinsman]


                return(genealogy)

    
    def __genealogy__(self, fidx):
        """
        Doc: 
            Calling Genealogy for Family 
            
        Args:
            fidx                                           hdf5 file

        Return: 
            genealogy_birthdates (np.arr):                 Birthdate of family members
            genealogy_degree (np.arr):                     Degree of familymebers
            genealogy_EOO (np.arr):                        End of Observation for family mebers
            genealogy_sex (np.arr):                        Sex of family mebers 
            genealogy_set (np.date):                       Cancer set - 
                                                           date - cancer - morph - degree - age - sex - index
            
        """
        
        G = self.__genealogy_extract__(fidx=fidx)
        
        genealogy_birthdates = np.concatenate([self.__array__(G[key][1], dtype='datetime64') for key in G.keys()])
        genealogy_degree = np.concatenate([self.__array__(G[key][2], dtype='float') for key in G.keys()])
        genealogy_EOO = np.concatenate([self.__array__(G[key][3], dtype='datetime64') for key in G.keys()])
        genealogy_sex = np.concatenate([self.__array__(G[key][4], dtype='str') for key in G.keys()])
        genealogy_kinsman = np.concatenate([self.__array__(G[key][5], dtype='str') for key in G.keys()])
        genealogy_set = np.concatenate([self.__darray__(G[key][0], d=7) for key in G.keys()])
        
        if len(G['c'][1]) > 0:
            genealogy_children = np.concatenate(G['c'][1])
        else:      
            genealogy_children = np.zeros((0,)).astype('datetime64[D]')
        genealogy_children = np.sort(genealogy_children)
        
        
        idx_min_degree = genealogy_degree >= 0.2 # only first&second degree
        genealogy_birthdates = genealogy_birthdates[idx_min_degree]
        genealogy_degree = genealogy_degree[idx_min_degree]
        genealogy_EOO = genealogy_EOO[idx_min_degree]
        genealogy_sex = genealogy_sex[idx_min_degree]
        genealogy_kinsman = genealogy_kinsman[idx_min_degree]
        genealogy_set = genealogy_set[genealogy_set[:, 3].astype(float) >= 0.2]
           
        
        return(genealogy_birthdates, genealogy_degree, genealogy_EOO, genealogy_sex, genealogy_set, genealogy_children, genealogy_kinsman)

    def __familyindicator__(self, genealogy_set):
        if genealogy_set.shape[0]>0: 
            family_cases = np.zeros((0, 20, 4))
            helpvar_dates = genealogy_set[:, 0].astype('datetime64[D]')
            gdates = []
            for tt in np.unique(helpvar_dates):
                gdates.extend([tt])
                idx_date = helpvar_dates <= tt
                n = 0
                gg = genealogy_set[idx_date, :].copy()
                helpvar = np.zeros((20, 4))
                for ii in self.genealogy_cancers:
                    idxc = np.max([(kk == gg[:, 1]).tolist() for kk in ii], axis=0) 
                    dd = gg[idxc, :].copy()
                    helpvar[n, :] = np.concatenate((
                    np.asarray(np.unique(dd[dd[:, 3].astype(float) > 0.4, -1]).shape[0] >=1).astype(int)[None], # case 1st degree
                    np.asarray(np.unique(dd[:, -1]).shape[0] >= 1).astype(int)[None], # case all
                    np.asarray(np.unique(dd[:, -1]).shape[0] >= 2).astype(int)[None], # multiple all 
                    np.asarray(np.unique(dd[dd[:, 4].astype(float) < 50*365, -1]).shape[0] >=1).astype(int)[None] # early age in family < 50
                ))
                    n +=1
                family_cases = np.concatenate((family_cases, helpvar[None, :, :]), axis=0)
            gdates = np.asarray(gdates)
        else: 
            family_cases = np.zeros((1, 20, 4))
            gdates = np.datetime64('1994-01-01')[None]

        return([gdates, family_cases])
    
    def __familypotential__(self, genealogy_birthdates, genealogy_EOO, genealogy_sex, SOO, EOO):
        if genealogy_birthdates[genealogy_birthdates <= EOO].shape[0] == 0:
            gdates = np.arange(SOO[0], EOO[0], np.timedelta64(365*5,'D'))
            family_risk = np.zeros((gdates.shape[0], 20))
        else:
            gdates = np.arange(SOO[0], EOO[0], np.timedelta64(365*5,'D'))
            family_age = np.floor((((genealogy_birthdates[:, None] - gdates) * -1) /365).astype(float)).astype(int)
            family_age = np.clip(family_age, -1, ((genealogy_EOO - genealogy_birthdates)/365)[:, None].astype(int))
            family_risk = np.zeros((gdates.shape[0], 20))

            # Add risk for each individual for each evaluation time
            for ind in range(family_age.shape[0]):
                for hh in range(len(gdates)):
                    family_risk[hh, :] += np.sum(self.nordcan[int(genealogy_sex[ind]), :, :((family_age[ind, hh]//5)+2)], axis=1)
                    
        return([gdates, np.log(family_risk+1)/10])
    
    
    def __genealogycall__(self, fidx, SOO, EOO, birthdate):
        
        # retrieve genealogy data
        genealogy_birthdates, genealogy_degree, genealogy_EOO, genealogy_sex, genealogy_set, genealogy_children, genealogy_kinsman = self.__genealogy__(fidx=fidx)
        genealogy_dates, family_cases = self.__familyindicator__(genealogy_set=genealogy_set)
        helpvar = (birthdate-genealogy_birthdates).astype(float)
        fmembers = np.sum(helpvar>0)/10
        fage = np.sum(helpvar[helpvar>0])/365/100
        
        helpvar = helpvar * (genealogy_degree >= 0.4).astype(float)
        fmembers_1st = np.sum(helpvar>0)/10
        fage_1st = np.sum(helpvar[helpvar>0])/365/100

        # sort
        idx_order = np.argsort(genealogy_dates)
        genealogy_dates = genealogy_dates[idx_order]
        family_cases = family_cases[idx_order]

        # collapse 1
        family_cases = np.asarray([np.sum(family_cases[jj == genealogy_dates, :, :], axis=0) for jj in np.unique(genealogy_dates)])
        
        # dates
        genealogy_dates = np.unique(genealogy_dates)

        return([genealogy_dates, family_cases, genealogy_children, [fmembers,
fage, fmembers_1st, fage_1st]])
   
        