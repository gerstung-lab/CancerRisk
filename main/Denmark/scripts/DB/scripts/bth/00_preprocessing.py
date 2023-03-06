# %%
# import modules

import sys 
import os 
import glob 
import h5py 
import datetime

import pandas as pd
import numpy as np 

np.random.seed(seed=83457)

dir_bth = '/home/people/alexwolf/data/BTH/raw/'
nrows=None

# %%
# main file
mapping = pd.read_csv('/home/people/alexwolf/data/BTH/raw/bth.csv', nrows=nrows)
mapping

# %%
bodymass = pd.read_csv('/home/people/alexwolf/data/BTH/raw/bodymassindex.csv', nrows=nrows)
bodymass

# %%
smoking = pd.read_csv('/home/people/alexwolf/data/BTH/raw/smoking.csv', nrows=nrows)
smoking

# %%
weight = pd.read_csv('/home/people/alexwolf/data/BTH/raw/weight.csv', nrows=nrows)
weight

# %%
pulse = pd.read_csv('/home/people/alexwolf/data/BTH/raw/pulse.csv', nrows=nrows)
pulse

# %%
pressure = pd.read_csv('/home/people/alexwolf/data/BTH/raw/bloodpressure.csv', nrows=nrows)
pressure

# %%
height = pd.read_csv('/home/people/alexwolf/data/BTH/raw/height.csv', nrows=nrows)
height

# %%
lpr = pd.read_csv('/home/people/alexwolf/data/BTH/raw/lpr2bth.tsv', sep='\t', nrows=nrows, header=None)
lpr

# %%
# merge all files onto main
mapping = mapping.merge(height, how='left', left_on='txt_id', right_on='entryid')
mapping.drop('entryid', axis=1, inplace=True)
mapping.drop('lineno', axis=1, inplace=True)

mapping = mapping.merge(weight, how='left', left_on='txt_id', right_on='entryid')
mapping.drop('entryid', axis=1, inplace=True)
mapping.drop('lineno', axis=1, inplace=True)

mapping = mapping.merge(pressure, how='left', left_on='txt_id', right_on='entryid')
mapping.drop('entryid', axis=1, inplace=True)
mapping.drop('lineno', axis=1, inplace=True)

mapping = mapping.merge(pulse, how='left', left_on='txt_id', right_on='entryid')
mapping.drop('entryid', axis=1, inplace=True)
mapping.drop('lineno', axis=1, inplace=True)

mapping = mapping.merge(smoking, how='left', left_on='txt_id', right_on='entryid')
mapping.drop('entryid', axis=1, inplace=True)
mapping.drop('lineno', axis=1, inplace=True)

mapping = mapping.merge(bodymass, how='left', left_on='txt_id', right_on='entryid')
mapping.drop('entryid', axis=1, inplace=True)
mapping.drop('lineno', axis=1, inplace=True)

mapping['alcohol_weekly'] = np.nan
mapping['alcohol'] = np.nan
mapping['smoking'] = np.nan

# %%
# additional extract
cleaned = pd.read_csv('/home/people/alexwolf/data/BTH/raw/cleaned-features-v2.tsv', sep='\t', nrows=nrows)
cleaned

# %%
# reformated the cleaned data
dd_cleaned = cleaned.loc[:, ['PID', 'REKVID', 'REKVDT']].copy()
dd_cleaned.columns=['pid', 'rekvid', 'rekvdt']
dd_cleaned['txt_id'] = ''

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'ALCOHOL_CATEGORY')] = np.nan
dd_cleaned['alcohol'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'ALCOHOL_WEEKLY')] = np.nan
dd_cleaned['alcohol_weekly'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'HEIGHT')] = np.nan
dd_cleaned['height'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'WEIGHT')] = np.nan
dd_cleaned['weight'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'BP_SYS')] = np.nan
dd_cleaned['systolic'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'BP_DIA')] = np.nan
dd_cleaned['diastolic'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'PULSE')] = np.nan
dd_cleaned['pulse'] = value

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'SMOKING_STATUS')] = np.nan
dd_cleaned['smoking'] = value
dd_cleaned['packyear'] = np.nan

value = np.asarray(cleaned['VALUE'].copy())
value[np.asarray(cleaned['KEY'] != 'BMI')] = np.nan
dd_cleaned['bmi'] = value

# %%
# combine data
mapping = mapping.append(dd_cleaned, ignore_index=True)

# %%
mapping

# %%
# remove entries without relevant information
idx = np.asarray(np.sum(pd.isnull(mapping.iloc[:, 4:]), axis=1) < 10)
mapping = mapping.loc[idx, :].reset_index(drop=True)

# %%
# remove double entries
mapping = mapping.groupby(['pid', 'rekvdt']).first().reset_index()

# %%
# merge with lpr information
mapping = mapping.merge(lpr, how='left', left_on='pid', right_on=1)

mapping.drop('rekvid', axis=1, inplace=True)
mapping.drop('txt_id', axis=1, inplace=True)
mapping.drop('pid', axis=1, inplace=True)
mapping.drop(1, axis=1, inplace=True)
mapping.rename(columns={0: 'pid'}, inplace=True)

mapping = mapping.loc[~pd.isnull(mapping['pid']), :]

# %%
# transform date
mapping.loc[:, 'rekvdt'] = mapping.loc[:, 'rekvdt'].apply(lambda x: datetime.datetime.strptime(str(x)[:-4], '%Y%m%d'))

# %%
# save
mapping.to_csv('/home/people/alexwolf/data/BTH/processed/bth.csv', sep=';')

# %%
exit()

# %%



