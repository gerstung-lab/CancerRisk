import sys
import os 
import tqdm
import h5py
import numpy as np 

import torch 
from torch.distributions import constraints
from torch.utils.data import Dataset, DataLoader, Sampler

import pyro 
import pyro.distributions as dist 
import pyro.poutine as poutine
from pyro.infer import Predictive, SVI, Trace_ELBO

dtype = torch.FloatTensor

np.random.seed(17)
torch.random.manual_seed(9)
pyro.set_rng_seed(22)

def predictor(data, level, detype=dtype):
    theta_dnpr = pyro.sample('theta_dnpr', dist.StudentT(1.5, loc=0, scale=0.01).expand([1, 1305])).type(dtype)
    theta_gene = pyro.sample('theta_gene', dist.StudentT(1.5, loc=0, scale=0.01).expand([1, 80])).type(dtype)
    theta_bth = pyro.sample('theta_bth', dist.StudentT(1.5, loc=0, scale=0.01).expand([1, 7])).type(dtype)
    
    pred = torch.mm(data[1], theta_dnpr.T) + torch.mm(data[2], theta_gene.T)  + torch.mm(data[3], theta_bth.T) 
    pyro.deterministic('pred', pred)
    return(pred)
    
    
    
    
    
    