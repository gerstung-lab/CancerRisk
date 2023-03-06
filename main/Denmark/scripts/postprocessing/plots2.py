# %%
# Modules
# ==========================================================================================
# ==========================================================================================
import sys
import os 
import h5py
import dill as pickle
import tqdm
import subprocess

import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import torch 
import pyro

# Custom Functions
sys.path.append('/users/projects/cancer_risk/main/scripts/ProbCox')
from _custom_functions import KM, CIF, metric_table, round_, quantile

# seeding
np.random.seed(11)
torch.random.manual_seed(12)
pyro.set_rng_seed(13)

# directories 
dir_data = '/users/projects/cancer_risk/data/'
dir_DB = '/users/projects/cancer_risk/data/DB/DB/raw/'
dir_out = '/users/projects/cancer_risk/main/output/'
dir_pred = '/users/projects/cancer_risk/data/predictions_ukb/'
dir_root = '/users/projects/cancer_risk/'

Events = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver',
                'Pancreas', 'Lung', 'Melanoma', 'Breast', 'Cervix uteri',
                'Corpus Uteri', 'Ovary', 'Prostate', 'Testis', 'Kidney',
                'Bladder', 'Brain', 'Thyroid', 'NHL', 'MM', 'AML', 'Other', 'Death']

events = ['oesophagus', 'stomach', 'colorectal', 'liver',
                'pancreas', 'lung', 'melanoma', 'breast', 'cervix_uteri',
                'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney',
                'bladder', 'brain', 'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

## Plotting Setup
#=======================================================================================================================

mpl.rcParams['axes.spines.left'] = True
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.grid'] = False

mpl.rcParams['axes.labelpad'] = 1
mpl.rcParams['axes.titlepad'] = 1
mpl.rcParams['xtick.major.pad'] = 1
mpl.rcParams['ytick.major.pad'] = 1
mpl.rcParams['hatch.linewidth'] = 0.5
plt.rcParams['font.size'] = 6

cm = 1/2.54

colormap= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#421a01', '#6e0b3c', 
                '#9370DB', '#7A378B', '#CD6090', '#006400', '#5ebd70', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CDCB50', '#CD6600', '#FF8C69', '#8f0000', '#b3b3b3', '#454545'])

run_id = int(sys.argv[1])
#run_id=0

# %%
# Models 
# ==========================================================================================
# ==========================================================================================
tt = [pickle.load(open(dir_root + 'main/model/' + events[cc] + '/param.pkl', 'rb')) for cc in range(22)]

A0 = []
for cc in tqdm.tqdm(range(22)):
    aa = pickle.load(open(dir_root + 'main/model/' + events[cc] + '/breslow.pkl', 'rb'))
    A0.extend([np.stack([[aa['female'](ii), aa['male'](ii)]for ii in range(31400)]).T])
A0 = np.stack(A0)

# %%
## Extract Data
#=======================================================================================================================
time = []
pred = []

file = dir_data + 'predictions_ukb/master_quick.h5'
with h5py.File(file, 'r') as f:
    time = f['time'][:]
    pred = f['pred'][:]
            
time = np.asarray(time).astype(float)
pred = np.asarray(pred).astype(float)

tt_surv = time[:, 0]
sex = time[:, 1].astype(bool)
age = time[:, 2].astype(int)
N = time.shape[0]
out = pred[:, 0, :].copy()
predage = np.arange(5840, 25581, 31)

#cc = run_id

# %%
prediction=[]
print(events[cc])
idx = np.logical_or(sex, ~sex)
ee = pred[idx, 0, cc].copy()
y_ = ee
tt_ = tt_surv[idx].copy()
cif_ = CIF(cc=cc, tt0=age[idx], tt_range=1825, A0=A0, pred=pred[idx, 1, :], sex=sex.astype(int)[idx])
pp=[]
for ii in tqdm.tqdm(range(np.sum(idx))):
    pp.extend(cif_(ii))
pp = np.asarray(pp)
prediction.extend([pp[:, None]])
prediction = np.concatenate((prediction), axis=1)
prediction = prediction[:, 0]

# %%
## 5 year Risk Distribution
#=======================================================================================================================
file = dir_data + 'predictions_ukb/master.h5'
with h5py.File(file, 'r') as ff:
    ll = ff['data'][:, :, cc]
    qq_f = np.zeros((637, 9))
    qq_m = np.zeros((637, 9))
    for jj in tqdm.tqdm(range(637)):
        if cc not in [11, 12]:
            idx_f = np.logical_and(ll[:, jj] > 0, ~sex)
            qq_f[jj, :] = quantile(ll[idx_f, jj], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

        if cc not in [7, 8, 9, 10]:
            idx_m = np.logical_and(ll[:, jj] > 0, sex)
            qq_m[jj, :] = quantile(ll[idx_m, jj], [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    # plot1 
    fig, ax = plt.subplots(1, 2, figsize=(10*cm, 4*cm), dpi=600, sharey=True)
    fig.subplots_adjust(wspace=0.05)
    ax[0].plot(predage/365, qq_f[:, 0], color=colormap[cc], lw=1, ls=(0, (1, 1)), label='1st quantile')
    ax[0].plot(predage/365, qq_f[:, 1], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)), label='5th quantile')
    ax[0].plot(predage/365, qq_f[:, 2], color=colormap[cc], lw=1, ls=(0, (5, 1)), label='10th quantile')
    ax[0].plot(predage/365, qq_f[:, 3], color=colormap[cc], lw=1, ls='-', label='25th quantile')

    ax[0].plot(predage/365, qq_f[:, 5], color=colormap[cc], lw=1, ls='-')
    ax[0].plot(predage/365, qq_f[:, 6], color=colormap[cc], lw=1, ls=(0, (5, 1)))
    ax[0].plot(predage/365, qq_f[:, 7], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)))
    ax[0].plot(predage/365, qq_f[:, 8], color=colormap[cc], lw=1, ls=(0, (1, 1)))
    ax[0].plot(predage/365, qq_f[:, 4], color='black', lw=1, ls='-', label='median')

    ax[1].plot(predage/365, qq_m[:, 0], color=colormap[cc], lw=1, ls=(0, (1, 1)))
    ax[1].plot(predage/365, qq_m[:, 1], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)))
    ax[1].plot(predage/365, qq_m[:, 2], color=colormap[cc], lw=1, ls=(0, (5, 1)))
    ax[1].plot(predage/365, qq_m[:, 3], color=colormap[cc], lw=1, ls='-')

    ax[1].plot(predage/365, qq_m[:, 5], color=colormap[cc], lw=1, ls='-')
    ax[1].plot(predage/365, qq_m[:, 6], color=colormap[cc], lw=1, ls=(0, (5, 1)))
    ax[1].plot(predage/365, qq_m[:, 7], color=colormap[cc], lw=1, ls=(0, (3, 1, 1, 1)))
    ax[1].plot(predage/365, qq_m[:, 8], color=colormap[cc], lw=1, ls=(0, (1, 1)))
    ax[1].plot(predage/365, qq_m[:, 4], color='black', lw=1, ls='-')

    ax[0].set_ylabel('5-year Risk')

    if cc in [7, 8, 9, 10, 16]:
        ax[0].legend(frameon=False, fontsize=5)
    else:
        ax[0].legend(frameon=False, fontsize=5)

    plt.savefig(dir_out + events[cc] + '/figures/5yr_risk_distribution.eps', dpi=600, bbox_inches='tight', transparent=True)
    plt.savefig(dir_out + events[cc] + '/figures/5yr_risk_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
    #plt.show()
    plt.close()

    dd = pd.DataFrame(np.concatenate((predage[:, None], qq_f, qq_m), axis=1))
    dd.columns =  ['age', 'q0.01_female', 'q0.05_female', 'q0.1_female', 'q0.25_female', 'q0.5_female', 'q0.75_female', 'q0.9_female', 'q0.95_female', 'q0.99_female', 'q0.01_male', 'q0.05_male', 'q0.1_male', 'q0.25_male', 'q0.5_male', 'q0.75_male', 'q0.9_male', 'q0.95_male', 'q0.99_male']
    dd.to_csv(dir_out + events[cc] + '/data/5yr_risk_distribution.csv')


# %%
## Absolute Risk Distribution
#=======================================================================================================================

if cc in [7, 8, 9, 10]:
    idx = ~sex
elif cc in [11, 12]:
    idx = sex
else:
    idx = np.logical_or(sex, ~sex)

qq = np.quantile(prediction[idx, 0], 0.99)
fig, ax = plt.subplots(1, 2, figsize=(7*cm, 3*cm), dpi=300, gridspec_kw={'width_ratios':[4, 1]}, sharey=True)
fig.subplots_adjust(wspace=0.1)

idxss = np.random.choice(np.where(np.logical_and(~out[:, cc].astype(bool), idx))[0], 50000)
ax[0].plot(age[idxss]/365, np.minimum(prediction[idxss, 0], qq), ls='', marker='x', markersize=0.75, markeredgewidth=0.05, color='.1')


try:
    idxss = np.random.choice(np.where(np.logical_and(out[:, cc].astype(bool), idx))[0], 2500)
except:
    idxss = np.logical_and(out[:, cc].astype(bool), idx)
    
ax[0].plot(age[idxss]/365, np.minimum(prediction[idxss, 0], qq), ls='', marker='x', markersize=1, markeredgewidth=0.25, color=colormap[cc])

ax[1].hist([np.minimum(prediction[idx, 0][~out[idx, cc].astype(bool)], qq), np.minimum(prediction[idx, 0][out[idx, cc].astype(bool)], qq)], color=['0.1', colormap[cc]], density=True, orientation="horizontal", bins=10, label=['Non-Cancer', 'Cancer'])

ax[0].set_ylabel('Absolute Risk 5yr.')
ax[0].set_xlabel('Age')
#ax[0].set_xticks([50, 60, 70])
#ax[0].set_ylim([0, 0.4])
ax[1].set_xlabel('Frequency')
#ax[1].legend(frameon=False, fontsize=5, bbox_to_anchor=(0.4, 0.85))
plt.savefig(dir_out + events[cc] + '/figures/absolute_risk_distribution.eps', dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(dir_out + events[cc] + '/figures/absolute_risk_distribution.pdf', dpi=600, bbox_inches='tight', transparent=True)
#plt.show()
plt.close()


# %%
## Screening
#=======================================================================================================================#
## Data Table

aa_ = []
base_n = []
base_c = []
base_age_mean = []
base_age_se = []
base_age_q10 = []
base_age_q25 = []
base_age_median = []
base_age_q75 = []
base_age_q90 = []

phs_n = []
phs_c = []
phs_age_mean = []
phs_age_se = []
phs_age_q10 = []
phs_age_q25 = []
phs_age_median = []
phs_age_q75 = []
phs_age_q90 = []

for aa in range(40, 66):
    aa_.extend([aa])
    # PPV for aa - aa+1 year range
    idxage_cut = np.logical_and(age/365 >= aa, age/365 < aa+5)
    idxage_early = age/365 < aa+5

    idx_early_female = np.logical_and(~sex, idxage_early)
    idx_early_male = np.logical_and(sex, idxage_early)

    n_f = (~sex[idxage_cut]).sum()
    n_m = sex[idxage_cut].sum()

    target_f = np.sort(prediction[idx_early_female])[-n_f]
    target_m = np.sort(prediction[idx_early_male])[-n_m]

    screen = np.zeros((prediction.shape[0],))

    if cc in [7, 8, 9, 10]:
        screen[np.logical_and(idx_early_female, prediction >= target_f)] = 1
        idxage_cut = np.logical_and(~sex, idxage_cut)
    elif cc in [10, 11]:
        screen[np.logical_and(idx_early_male, prediction >= target_m)] = 1
        idxage_cut = np.logical_and(sex, idxage_cut)
    else:
        screen[np.logical_and(idx_early_female, prediction >= target_f)] = 1
        screen[np.logical_and(idx_early_male, prediction >= target_m)] = 1

    screen = screen.astype(bool)

    #bdd1.extend([age[np.logical_and(idxage_cut, out[:, cc]==1)]/365]) 
    #bdd2.extend([age[np.logical_and(screen, out[:, cc]==1)]/365]) 

    base_n.extend([idxage_cut.sum()])
    base_c.extend([np.logical_and(idxage_cut, out[:, cc]==1).sum()])
    base_age_mean.extend([np.mean(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365)])
    base_age_se.extend([np.sqrt(np.var(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365))])
    base_age_q10.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.1)])
    base_age_q25.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.25)])
    base_age_median.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.5)])
    base_age_q75.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.75)])
    base_age_q90.extend([np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.9)])

    phs_n.extend([screen.sum()])
    phs_c.extend([np.logical_and(screen, out[:, cc]==1).sum()])
    phs_age_mean.extend([np.mean(age[np.logical_and(screen, out[:, cc]==1)]/365)])
    phs_age_se.extend([np.sqrt(np.var(age[np.logical_and(screen, out[:, cc]==1)]/365))])
    phs_age_q10.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.1)])
    phs_age_q25.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.25)])
    phs_age_median.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.5)])
    phs_age_q75.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.75)])
    phs_age_q90.extend([np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.9)])

aa_ = np.asarray(aa_)
base_n = np.asarray(base_n)
base_c = np.asarray(base_c)
base_age_mean = np.asarray(base_age_mean)
base_age_se = np.asarray(base_age_se)
base_age_q10 = np.asarray(base_age_q10)
base_age_q25 = np.asarray(base_age_q25)
base_age_median = np.asarray(base_age_median)
base_age_q75 = np.asarray(base_age_q75)
base_age_q90 = np.asarray(base_age_q90)
phs_n = np.asarray(phs_n)
phs_c = np.asarray(phs_c)
phs_age_mean = np.asarray(phs_age_mean)
phs_age_se = np.asarray(phs_age_se)
phs_age_q10 = np.asarray(phs_age_q10)
phs_age_q25 = np.asarray(phs_age_q25)
phs_age_median = np.asarray(phs_age_median)
phs_age_q75 = np.asarray(phs_age_q75)
phs_age_q90 = np.asarray(phs_age_q90)

dd = pd.DataFrame(np.concatenate((aa_[:, None],
    base_n[:, None],
    base_c[:, None],
    base_age_mean[:, None],
    base_age_se[:, None],
    base_age_q10[:, None],
    base_age_q25[:, None],
    base_age_median[:, None],
    base_age_q75[:, None],
    base_age_q90[:, None],
    phs_n[:, None],
    phs_c[:, None],
    phs_age_mean[:, None],
    phs_age_se[:, None],
    phs_age_q10[:, None],
    phs_age_q25[:, None],
    phs_age_median[:, None],
    phs_age_q75[:, None],
    phs_age_q90[:, None]
), axis=1))
dd.columns = ['age_threshold', 'base_n', 'base_c', 'base_age_mean', 'base_age_se', 'base_age_q10', 'base_age_q25', 'base_age_median', 'base_age_q75', 'base_age_q90', 'phs_n', 'phs_c', 'phs_age_mean', 'phs_age_se', 'phs_age_q10', 'phs_age_q25', 'phs_age_median', 'phs_age_q75', 'phs_age_q90', ]
dd.to_csv(dir_out + events[cc] + '/data/screening.csv')  

## 
bdd1 = []
bdd2 = []
aa = 55
# PPV for aa - aa+1 year range
idxage_cut = np.logical_and(age/365 >= aa, age/365 < aa+5)
idxage_early = age/365 < aa+5

idx_early_female = np.logical_and(~sex, idxage_early)
idx_early_male = np.logical_and(sex, idxage_early)

n_f = (~sex[idxage_cut]).sum()
n_m = sex[idxage_cut].sum()

target_f = np.sort(prediction[idx_early_female])[-n_f]
target_m = np.sort(prediction[idx_early_male])[-n_m]

screen = np.zeros((prediction.shape[0],))

if cc in [7, 8, 9, 10]:
    screen[np.logical_and(idx_early_female, prediction >= target_f)] = 1
    idxage_cut = np.logical_and(~sex, idxage_cut)
elif cc in [10, 11]:
    screen[np.logical_and(idx_early_male, prediction >= target_m)] = 1
    idxage_cut = np.logical_and(sex, idxage_cut)
else:
    screen[np.logical_and(idx_early_female, prediction >= target_f)] = 1
    screen[np.logical_and(idx_early_male, prediction >= target_m)] = 1

screen = screen.astype(bool)

np.save(dir_out + events[cc] + '/data/bdd1.npy', age[np.logical_and(idxage_cut, out[:, cc]==1)]/365)
np.save(dir_out + events[cc] + '/data/bdd2.npy', age[np.logical_and(screen, out[:, cc]==1)]/365) 

print(events[cc])
print(Events[cc], np.round(np.logical_and(screen, out[:, cc]==1).sum()/np.logical_and(idxage_cut, out[:, cc]==1).sum(), 2))
print(np.round(np.quantile(age[np.logical_and(screen, out[:, cc]==1)]/365, 0.25) - np.quantile(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365, 0.25) , 2))
print(np.round(np.mean(age[np.logical_and(screen, out[:, cc]==1)]/365) - np.mean(age[np.logical_and(idxage_cut, out[:, cc]==1)]/365) , 2))

# %%
print('finished')
exit()


# %%
%%bash

rm run.sh

echo '
#!/bin/sh
#PBS -N plots
#PBS -o /users/projects/cancer_risk/_/
#PBS -e /users/projects/cancer_risk/_/
#PBS -l nodes=1:ppn=1
#PBS -l mem=16gb
#PBS -l walltime=12:00:00

cd $PBS_O_WORDIR
module load anaconda3/2019.10
source conda activate
module load tools
module load gcc/10.2.0
module load intel/perflibs/2018
module load R/4.1.0

jupyter nbconvert --to script /users/projects/cancer_risk/main/scripts/postprocessing/plots2.ipynb --output /users/projects/cancer_risk/main/scripts/postprocessing/plots2

/services/tools/anaconda3/2019.10/bin/python3.7 /users/projects/cancer_risk/main/scripts/postprocessing/plots2.py $VAR1
' >> run.sh

for ii in {0..19}; do qsub -v VAR1=$ii run.sh; done

# %%
# Figure
bdd1 = []
bdd2 = []
for cc in range(20):
    if cc not in [6, 8, 12, 16]:
        bdd1.extend([np.load(dir_out + events[cc] + '/data/bdd1.npy')])
        bdd2.extend([np.load(dir_out + events[cc] + '/data/bdd2.npy')])
        
Events_ = ['Oesophagus', 'Stomach', 'Colorectal', 'Liver', 'Pancreas', 'Lung', 'Breast', 
                 'Corpus Uteri', 'Ovary', 'Prostate', 'Kidney', 'Bladder', 'Brain',
                'NHL', 'MM', 'AML']

colormap_= np.asarray(['#1E90FF', '#BFEFFF', '#191970', '#87CEFA', '#008B8B', '#946448', '#6e0b3c', 
                 '#7A378B', '#CD6090', '#006400', '#f8d64f', '#EEAD0E', '#f8d6cf',
                '#CD6600', '#FF8C69', '#8f0000'])

fig, ax = plt.subplots(1, 1, figsize=(15*cm, 4*cm), dpi=300)

bplot1 = ax.boxplot(bdd1, notch=False,
                     vert=True, showfliers=False,
                     patch_artist=True, medianprops=dict(color='black', lw=1), boxprops=dict(lw=0.3, hatch='xxxxxxxx'), whiskerprops=dict(lw=0.5), positions=np.arange(16), widths=0.35,  capprops=dict(lw=0.5))


bplot2 = ax.boxplot(bdd2, notch=False,
                     vert=True, showfliers=False,
                     patch_artist=True, medianprops=dict(color='black', lw=1), boxprops=dict(lw=0.3), whiskerprops=dict(lw=0.5), positions=np.arange(16)+0.4, widths=0.35,  capprops=dict(lw=0.5))

# fill with colors
for patch, color in zip(bplot1['boxes'], colormap_):
    patch.set_facecolor(color)

for patch, color in zip(bplot2['boxes'], colormap_):
    patch.set_facecolor(color)

    
#ax.set_ylim([40, 66])
ax.set_xticks(np.arange(0, 16)+0.2)
ax.set_xticklabels(np.asarray(Events_), rotation=90)

ax.axhline(60, color='black', lw=0.75)
ax.axhline(55, color='black', lw=0.75)
plt.savefig(dir_out + 'main' + '/figures/screening.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()   

# Summary Table
dd = pd.read_csv(dir_out + events[cc] + '/data/screening.csv')
dd = dd.iloc[[0, 5, 10, 15, 20, 25], :]
dd.reset_index(inplace=True, drop=True)
dd.iloc[:, 0] = np.repeat(Events[0], 6)

for cc in range(1, 20):
    dd_ = pd.read_csv(dir_out + events[cc] + '/data/screening.csv')
    dd_ = dd_.iloc[[0, 5, 10, 15, 20, 25], :]
    dd_.reset_index(inplace=True, drop=True)
    dd_.iloc[:, 0] = np.repeat(Events[cc], 6)
    dd = dd.append(dd_)
    
dd.to_csv(dir_out + events[cc] + '/tables/screening.csv')


# %%
# Summary Table
dd = pd.read_csv(dir_out + events[cc] + '/data/screening.csv')
dd = dd.iloc[[15], :]
dd.reset_index(inplace=True, drop=True)
dd.iloc[:, 0] = np.repeat(Events[0], 1)

for cc in range(1, 20):
    if cc not in [6, 8, 12, 16]:
        dd_ = pd.read_csv(dir_out + events[cc] + '/data/screening.csv')
        dd_ = dd_.iloc[[15], :]
        dd_.reset_index(inplace=True, drop=True)
        dd_.iloc[:, 0] = np.repeat(Events[cc], 1)
        dd = dd.append(dd_)
    
    
    
phs_cases = np.asarray(dd['phs_c'])
baseline_cases = np.asarray(dd['base_c'])
rr = phs_cases/baseline_cases
n = np.asarray(dd['phs_n'])
rr_l, rr_u = np.exp(np.log(phs_cases/baseline_cases) - np.sqrt((n-phs_cases)/phs_cases/n + (n-baseline_cases)/baseline_cases/n)), np.exp(np.log(phs_cases/baseline_cases) + np.sqrt((n-phs_cases)/phs_cases/n + (n-baseline_cases)/baseline_cases/n))

mpl.rcParams['axes.spines.bottom'] = False
fig, ax = plt.subplots(1, 1, figsize=(15*cm, 1*cm), dpi=300)

ax.bar(range(16), rr-1, color=colormap_, bottom=1, width=0.5)

for cc in range(16):
    ax.plot([cc, cc], [rr_l[cc], rr_u[cc]], lw=0.75,  color='black')


ax.axhline(1, lw=0.5, color='black')
#ax.set_ylim([0.5, 1.5])
ax.set_xticks(np.arange(0, 16))
ax.set_xlim([-0.75, 15.6])
ax.set_xticklabels(np.asarray(Events_), rotation=90)
ax.get_xaxis().set_visible(False)
plt.savefig(dir_out + 'main' + '/figures/screening2.pdf', dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close() 



# %%
dd

# %%



