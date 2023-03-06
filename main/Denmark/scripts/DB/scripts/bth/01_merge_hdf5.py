# %%
# import modules
print('start')
import sys 
import os 
import tqdm
import glob 
import h5py 
import datetime

import pandas as pd
import numpy as np 

np.random.seed(seed=83457)

dir_bth = '/home/people/alexwolf/data/BTH/processed/'

dir_data = '/home/people/alexwolf/data/DB/DB/raw/'


i = (int(sys.argv[1]))
j = (int(sys.argv[2]))

print(str(i), str(j))

# %%
# load BTH data
mapping = pd.read_csv('/home/people/alexwolf/data/BTH/processed/bth.csv', sep=';')
mapping = mapping.astype(str)
mapping.replace({'nan':''})
mapping.drop('Unnamed: 0', axis=1, inplace=True)
mapping.set_index('pid', inplace=True)


# %%
mapping = pd.read_csv('/home/people/alexwolf/data/BTH/processed/bth.csv', sep=';', nrows=None, usecols=[9])

# %%
# load files for cluster
files = glob.glob(dir_data + 'f_' + str(i) + '/f_' + str(j) + '/*')

# %%
# loop through ids and add bth data
for ff in files:
    with h5py.File(ff, 'a') as f:
        idx_list = list(f.keys())
        for ii in tqdm.tqdm(idx_list):
            try:
                del f[ii]['bth']
            except:
                pass
            try:
                dd = np.asarray(mapping.loc[ii].copy()).astype('S10')
                if len(dd.shape) == 1: 
                    dd = dd[None, :]
            except:
                dd = np.repeat('', 11)[None, :]
            f[ii].create_dataset('bth', data=np.asarray(dd).astype('S10'), maxshape=(None, 11), compression="lzf") 

# %%
print('finished')
exit()

# %% [markdown]
# # Cluster:

# %%
%%bash

rm /home/people/alexwolf/run.sh
rm /home/people/alexwolf/projects/ProbCox/script/01_merge_hdf5.py

ssh precision05
echo '
#!/bin/sh
#PBS -N merge_hdf5
#PBS -o /home/people/alexwolf/_/
#PBS -e /home/people/alexwolf/_/
#PBS -l nodes=1:ppn=2
#PBS -l mem=12gb
#PBS -l walltime=24:00:00

cd $PBS_O_WORDIR
module load tools
module load anaconda3/5.3.0
source conda activate

script_name=01_merge_hdf5

jupyter nbconvert --to script /home/people/alexwolf/projects/BTH/scripts/$script_name.ipynb --output /home/people/alexwolf/projects/BTH/scripts/$script_name

/services/tools/anaconda3/5.3.0/bin/python3.6 /home/people/alexwolf/projects/BTH/scripts/$script_name.py $VAR1 $VAR2
' >> run.sh

for ii in {6..9}; do for jj in {0..9}; do qsub -v VAR1=$ii,VAR2=$jj run.sh; done; done


# %%



