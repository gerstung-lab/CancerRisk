import os 
import sys
import h5py 
import torch
import numpy as np

class pipe_main():
    def __init__(self, **kwds):
        """
        Doc: 
            Main Pipe
            - extract baseline info on individual

            
        """
        super().__init__(**kwds)  
    
    def __maincall__(self, fidx):
        """
        Doc: 
            Retrieves baseline information contained in the HDF5 for individual idx

        Args:
            fidx:                   hdf5 file

        Return: 
            birthdate (np.datetime)(1,): Date of birth
            sex (np.int)(1,):            Sex
            status (np.int)(1,):         Status of individual - alive/dead/moved
            EOO (np.datetime)(1,):       End of Observation
            idxcancer (np.int)(1,):      Indicator whether cancer incidence
        """
        
        birthdate = fidx.attrs['birthdate'].astype('datetime64')
        SOO = np.maximum(birthdate, np.datetime64('1994-01-01'))
        sex = fidx.attrs['sex']
        status = fidx.attrs['status']
        EOO = fidx.attrs['EOO'].astype('datetime64')
        idxcancer = fidx.attrs['cancer']
        return(birthdate, sex, status, EOO, SOO, idxcancer)

