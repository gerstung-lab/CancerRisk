# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import pickle
import dill
import tqdm
import shutil

import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
import pyro.poutine as poutine
import pyro.poutine as poutine
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer.autoguide import AutoLowRankMultivariateNormal
pyro.clear_param_store()

import warnings
warnings.filterwarnings("ignore")

# model - ProbCox
sys.path.append('/users/secureome/home/people/alexwolf/projects/CancerRisk/scripts/ProbCox')
import probcox as pcox 

# predictor
from m1 import predictor 

# dataloader
sys.path.append('/users/secureome/home/people/alexwolf/projects/CancerRisk/scripts/dataloader')
from dataloader import Data_Pipeline

# help functions
from helper import custom_collate, IterativeSampler, RandomSampler

# seeding
np.random.seed(739)
torch.random.manual_seed(587)
pyro.set_rng_seed(230)

# directories 
dir_data = '/users/secureome/home/people/alexwolf/data/'
dir_DB = '/users/secureome/home/people/alexwolf/data/DB/DB/raw/'
dir_out = '/users/secureome/home/people/alexwolf/projects/CancerRisk/output/'
dir_logs = '/users/secureome/home/people/alexwolf/projects/CancerRisk/logs/'

# train/valid/test
with open(dir_DB + 'trainvalidtest.pickle', 'rb') as handle:
    tvt_split = pickle.load(handle)
    
# proportions
with open(dir_data + 'CancerRisk/proportions.pickle', 'rb') as handle:
    proportions = pickle.load(handle)
    
sampling_proportion = []
for cc in range(7):
    sampling_proportion.append([proportions[0]+proportions[2], 720, proportions[1][cc]+proportions[3][cc], None])
for cc in range(7, 11):
    sampling_proportion.append([proportions[0], 720, proportions[1][cc], None])
for cc in range(7, 9):
    sampling_proportion.append([proportions[2], 720, proportions[3][cc], None])
for cc in range(11, 20):
    sampling_proportion.append([proportions[0]+proportions[2], 720, proportions[1][cc]+proportions[3][cc-2], None])

dtype = torch.FloatTensor 

#event_idx = int(sys.argv[1])
#event_idx=20

np.random.seed(event_idx+43)
torch.random.manual_seed(event_idx+46)
pyro.set_rng_seed(event_idx+89)

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

# %%
try:
    shutil.rmtree(dir_out + 'model/' + events[event_idx])
except:
    pass
os.mkdir(dir_out + 'model/' + events[event_idx])

# %%
# Data Pipeline
# ==========================================================================================
# ==========================================================================================
sampling = RandomSampler(ids=tvt_split['train'], iteri=10011, unique=8) # empty sampler

if event_idx in [7, 8, 9, 10]:
    PIPE = Data_Pipeline(event_idx=event_idx, sex_specific=np.asarray([0]))
elif event_idx in [11, 12]:
    PIPE = Data_Pipeline(event_idx=event_idx, sex_specific=np.asarray([1]))  
else:
    PIPE = Data_Pipeline(event_idx=event_idx, sex_specific=np.asarray([0, 1]))

dataloader = DataLoader(PIPE, batch_size=1, num_workers=17, prefetch_factor=1, collate_fn=custom_collate, sampler=sampling)

# %%
# Inference
# ==========================================================================================
# ==========================================================================================
loss=[0]
pyro.clear_param_store()

# Guide 
m = pcox.PCox(sampling_proportion=sampling_proportion, predictor=predictor, guide=None, levels=event_idx)
#m.initialize(eta=0.000004, rank=50) 
m.initialize(eta=0.0000001, rank=50) 
cc = []
n=0
iteri=0
data = [[], [], [], [], []]
for _, __input__ in (enumerate(dataloader)):
    n+=1
    [data[kk].extend(__input__[kk]) for kk in range(4)]
    if n == 8:
        iteri+=1
        
        print(iteri)
        n = 0
        data[0] = torch.tensor(data[0]).type(dtype)
        data[1] = torch.tensor(data[1]).type(dtype)
        data[2] = torch.tensor(data[2]).type(dtype)
        data[3] = torch.tensor(data[3]).type(dtype)
        
        # save  
        if iteri % 250 == 0:
            with torch.no_grad():
                gg = m.return_guide()
                mm = m.return_model()
                pyro.get_param_store().save(dir_out + 'model/' + events[event_idx] +'/model.pkl')
                dill.dump({'model':mm, 'guide':gg}, open(dir_out + 'model/' + events[event_idx] + '/param.pkl', 'wb'))
                with open(dir_logs + events[event_idx] +'.txt', 'w') as ww:
                    ww.write(str(iteri))
                    
                predictive = Predictive(model=mm, guide=gg, num_samples=10, return_sites=(['pred']))
                samples = predictive(data)
                pred = torch.mean(torch.squeeze(samples['pred']), axis=0)[:, None].numpy()
                plt.hist(pred)
                plt.show()
                plt.close()

                cc.extend([pcox.concordance(surv=data[0].detach().numpy(), predictions=pred)])
                fig, ax = plt.subplots(1, 1, figsize=(8.27, 11.69/2), dpi=300)
                ax.plot(cc)
                plt.savefig(dir_logs + events[event_idx] +'.png', bbox_inches='tight')
                plt.close()
        
        # reset      
        loss.append(m.infer(data=data))
        if iteri % 1800==0:
            m.decay()
        
        data = [[], [], [], [], []]
        

# %%
exit()

# %%
%%bash

rm /home/people/alexwolf/run.sh

ssh precision05
echo '
#!/bin/sh
#PBS -N cox
#PBS -o /home/people/alexwolf/_/
#PBS -e /home/people/alexwolf/_/
#PBS -l nodes=1:ppn=13
#PBS -l mem=18gb
#PBS -l walltime=300:00:00

cd $PBS_O_WORDIR
module load tools
module load anaconda3/5.3.0
source conda activate

jupyter nbconvert --to script /home/people/alexwolf/projects/CancerRisk/scripts/m1/fit.ipynb --output /home/people/alexwolf/projects/CancerRisk/scripts/m1/fit

/services/tools/anaconda3/5.3.0/bin/python3.6 /home/people/alexwolf/projects/CancerRisk/scripts/m1/fit.py $VAR1
' >> run.sh


for ii in 20 21; do sleep 10; qsub -v VAR1=$ii run.sh; done



# %%



