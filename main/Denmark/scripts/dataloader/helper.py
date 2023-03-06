import os 
import sys
import h5py 
import torch
import pyro
import numpy as np 
from itertools import chain
from torch.utils.data.sampler import Sampler

import matplotlib as mpl
import matplotlib.pyplot as plt

dtype = torch.FloatTensor 

# Samnpler
# =======================================================================================================================   

# Simple random sampler
# =======================================================================================================================
class RandomSampler(Sampler):
    def __init__(self, ids, iteri=None, unique=16):
        self.ids = ids
        self.unique = unique
        if iteri:
            self.iteri = iteri
        else: 
            self.iteri = 10e100

    def __iter__(self):
        for __ in range(self.iteri):
            for _ in np.random.choice(self.ids, self.unique, replace=False):
                yield(_)

    def __len__(self):
        return self.iteri

# Simple iterative sampler
# =======================================================================================================================
class IterativeSampler(Sampler):
    def __init__(self, ids):
        self.ids_len = len(ids)
        self.ids = ids

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return self.ids_len
    
# custom tensor transform
# =======================================================================================================================   
def to_tensor(res, dtype=dtype):
    return([[torch.tensor(res[0][ii]).type(dtype) for ii in range(38)],
       torch.tensor(res[1]).type(dtype), 
       torch.tensor(res[2]).type(dtype), 
       torch.tensor(res[3]).type(dtype),
       torch.tensor(res[4]).type(dtype),
       torch.tensor(res[5]).type(dtype)])

# custom collate for torch dataloader
# =======================================================================================================================
def custom_collate(batch):
    return(batch[0])
    