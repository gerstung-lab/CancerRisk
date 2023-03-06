## Modules
#=======================================================================================================================
import os 
import sys 
import shutil
import tqdm
import pickle
import dill
import h5py
import torch 
import pyro 

import probcox as pcox
import numpy as np 
import pandas as pd 

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

## Main
#=======================================================================================================================

ukb_iterator = pd.read_csv(ROOT_DIR + 'main/45632/ukb45632.csv', usecols=[0]) # iterate over ukb dataset for 5000 per job

ukb_iterator_add = pd.read_csv(ROOT_DIR + 'main/44968/ukb44968.csv', usecols=[0]) # iterate over ukb dataset for 5000 per job

ll = []
offset = 0
for ii in tqdm.tqdm(range(ukb_iterator.shape[0])):
    if ukb_iterator.iloc[ii, 0] != ukb_iterator_add.iloc[ii+offset, 0]:
        ll.append(ukb_iterator_add.iloc[ii+offset, 0])
        offset+=1

with open(ROOT_DIR + 'main/44968/ukb44968s.csv','w') as tmp:
    with open(ROOT_DIR + 'main/44968/ukb44968.csv', 'r') as infile:
        for linenumber, line in tqdm.tqdm(enumerate(infile)):
            if linenumber > 0:
                if int(line.split('"')[1]) not in ll:
                    tmp.write(line)
            else:
                tmp.write(line)
                

ukb_iterator_add = pd.read_csv(ROOT_DIR + 'main/44968/ukb44968s.csv', usecols=[0]) # iterate over ukb dataset for 5000 per job

ukb_iterator.shape
ukb_iterator_add.shape

ukb_iterator.iloc[:, 0] == ukb_iterator_add.iloc[:, 0]

np.sum(ukb_iterator.iloc[:, 0] != ukb_iterator_add.iloc[:, 0])




