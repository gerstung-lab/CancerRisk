# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import pickle
import tqdm

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'
dir_tmp = '/users/projects/cancer_risk/tmp/'

# train/valid/test
with open(dir_DB + 'trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)

dsplit='train'

event_idx = int(sys.argv[1])

# %%
time = []
pred = []
for run_id in tqdm.tqdm(tvt_split[dsplit]):
    ii1 = run_id//100
    ii2 = (run_id - ii1*100)//10
    file = dir_data + 'predictions/f_%i/f_%i/_%i.h5' %(ii1, ii2, run_id)
    with h5py.File(file, 'r') as f:
        time.extend(f['main'][:, (2, 3, 6)].tolist())
        pred.extend(f['pred'][:, (0, 2, 3, 4, 5), event_idx].tolist())

time = np.asarray(time)
pred = np.asarray(pred)
sex = time[:, -1].astype(bool)
time = time[:, :-1]

# %%
pd.DataFrame(np.concatenate((time, pred, sex[:, None]), axis=1)).to_csv(dir_tmp + str(dsplit) + '_' + str(event_idx) + '.csv', sep=';')

# %%
exit()

# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N concordance
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=2
#PBS -l mem=16gb
#PBS -l walltime=48:00:00

cd $PBS_O_WORDIR
module load anaconda3/2019.10
source conda activate

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/m1/concordance.ipynb --output /users/projects/cancer_risk/main/scripts/m1/concordance

/services/tools/anaconda3/2019.10/bin/python3.7 /users/projects/cancer_risk/main/scripts/m1/concordance.py $VAR1
' >> run.sh

for ii in {0..21}; do qsub -v VAR1=$ii run.sh; done


# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N concordance_ev
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=2
#PBS -l mem=32gb
#PBS -l walltime=48:00:00

cd $PBS_O_WORDIR
module load gcc/10.2.0
module load tools
module load intel/perflibs/2018
module load R/4.1.0

Rscript /users/projects/cancer_risk/main/scripts/m1/concordance.R --no-save --no-restore $VAR1
' >> run.sh

for ii in {0..21}; do qsub -v VAR1=$ii run.sh; done


# %%



