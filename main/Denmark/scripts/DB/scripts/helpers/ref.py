# loading packages
import pandas as pd
import numpy as np 
import h5py
import sys
import pickle 

# path
d_data = '/home/people/alexwolf/data/'
d_dnpr = '/home/projects/registries/2018/classic_style_lpr'

# load cancer data 
keys = []
values = []
for ind in np.arange(1000):
    print(ind)
    with h5py.File(d_data + 'DB/DB/raw/_' + str(ind), 'r') as f:
        for idx in list(f.keys()):
            keys.append(idx)
            values.append([f[idx].attrs['sex'], f[idx].attrs['birthdate'], f[idx]['cancer']['set'][:]])

cancer_ref = dict(zip(keys, values))

f = open(d_data + 'DB/genealogy/_ref.pkl',"wb")
pickle.dump(cancer_ref, f)
f.close()

print('finsihed')

