'''
Pipe for extracting and making a decision on cancers
'''

# loading packages
import sys
import os 
import h5py
import numpy as np 

sys.path.append('/users/projects/cancer_risk/main/scripts/dataloader')
from pipe_main import pipe_main

class pipe_helper(pipe_main):
    def __init__(self, **kwds):
        """
        Doc: 
            - Functions to supplement the other pipelines

            -
        """
        
        super().__init__(**kwds)
        
    def __in__(self, x, y, reduce_x=False):
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

    def __array__(self, x, dtype):
        """
        Doc: 
            Transforms list to array and reshapes it to (:,)
            Helper function for Genealogy

        Args:
            x (list):                    list with entries from genealogy extraction
            dtype (str):                 dtype of final numpy array

        Return: 
            x array(dtype)(:,)
        """
        if len(x) > 1: 
            x = np.concatenate(x, axis=0)
        else: 
            x = np.asarray(x)
        try:
            a, b = x.shape
        except:
            a = x.shape[0]
            b = 1
        return(np.reshape(np.asarray(x).astype(dtype), (a*b,)))
    
    def __darray__(self, x, d=7):
        """
        Doc: 
            Transforms list to d-dimensional array
            Helper function for Genealogy

        Args:
            x (list):                    list with entries from genealogy extraction
            d (int):                     number of dimensions

        Return: 
            x array)(:, d)
        """
        try:
            x.shape[1] == d
        except:
            try: 
                x = np.concatenate(x)
            except: 
                x = np.zeros((0, d))[[]]
        return(x)
    






        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        