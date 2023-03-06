'''
Dataloader
'''
# Modules
#=======================================================================================================================
import tqdm
import h5py
import numpy as np

import pyro
from pyro.infer import Predictive

import torch
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader, Sampler

np.random.seed(7)

ROOT_DIR = '/nfs/research1/gerstung/sds/sds-ukb-cancer/'

dtype = torch.FloatTensor
# Sampler
#=======================================================================================================================
class RandomSampler(Sampler):
    def __init__(self, idx, iteri):
        self.idx = idx
        self.iteri = iteri

    def __iter__(self):
        for __ in range(self.iteri):
            for _ in np.random.choice(self.idx, len(self.idx), replace=False):
                yield(_)

    def __len__(self):
        return len(self.idx) * self.iteri
    
    
class ProportionalSampler(Sampler):
    def __init__(self, idx, eventlocations, iteri, eventsamples, randomsamples, batchsize):
        self.idx = np.asarray(idx).astype(int)
        self.iteri = np.asarray(iteri).astype(int)
        self.eventlocations = np.asarray(eventlocations).astype(int)
        self.eventsamples = np.asarray(eventsamples).astype(int)
        self.randomsamples = np.asarray(randomsamples).astype(int)
        self.batchsize = np.asarray(batchsize).astype(int)

    def __iter__(self):
        for __ in range(self.iteri):
            a, b = np.unique(np.concatenate((np.random.choice(self.eventlocations, self.eventsamples, replace=False), np.random.choice(self.idx, self.randomsamples, replace=False))), return_index=True)
            a = a[np.argsort(b)]
            for _ in a[:self.batchsize]:
                yield(str(_))

    def __len__(self):
        return len(self.idx) * self.iteri


# Datalaoder
#=======================================================================================================================
class PIPE(Dataset):
    def __init__(self, file, event_idx):
        """
        """
        self.f = h5py.File(file, 'r')
        self.event_idx = event_idx
        
    def __len__(self):
        return(10e100)

    def __getitem__(self, eid):
        time = self.f[eid]['MedRec']['time'][:]
        X = np.zeros(self.f[eid]['MedRec']['dim'][:])
        X[self.f[eid]['MedRec']['row_ind'][:], self.f[eid]['MedRec']['col_ind'][:]] = 1
        X = np.minimum(1, np.cumsum(X, axis=0))[:-1, :]                       
        idx_event = X[:, self.event_idx] == 1
        time =  time[~idx_event, :]   
        X =  X[~idx_event, :] 
        X = np.delete(X, self.event_idx, axis=1)
        if np.sum(idx_event) > 0:
            time[-1, -1] = 1
        return(torch.from_numpy(time).type(dtype), torch.from_numpy(X).type(dtype))
                       
    def __close__(self):
        self.f.close()
                       
                       
# Datalaoder
#=======================================================================================================================
def custom_collate(batch):
    return([torch.cat([item[ii] for item in batch], 0) for ii in range(len(batch[0]))])
