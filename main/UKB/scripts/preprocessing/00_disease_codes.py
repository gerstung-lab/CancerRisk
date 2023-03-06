import sys
import os
import tqdm
import h5py
import pickle
import numpy as np
import pandas as pd

np.random.seed(7)

ROOT_DIR = '/nfs/research/sds/sds-ukb-cancer/'

disease_codes = pd.read_csv(ROOT_DIR + 'projects/CancerRisk/data/raw/coding19.tsv', nrows=None, usecols=[0, 1, 3], sep='\t')
disease_codes = disease_codes.loc[disease_codes['coding'].apply(lambda x: len(x) == 3)]
disease_codes
idx_keep = disease_codes['coding'].apply(lambda x: x[0] not in ['S', 'T', 'U', 'V', 'W', 'X', 'Y' , 'Z', 'C'])
disease_codes = disease_codes[idx_keep]
disease_codes = disease_codes.reset_index(drop=True)

disease_codes.loc[:172, 'parent_id'] = 'Certain infectious and parasitic diseases'
disease_codes.loc[173:220, 'parent_id'] = 'Neoplasms'
disease_codes.loc[221:254, 'parent_id'] = 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism'
disease_codes.loc[255:327, 'parent_id'] = 'Endocrine, nutritional and metabolic diseases'
disease_codes.loc[328:405, 'parent_id'] = 'Mental and behavioural disorders'
disease_codes.loc[406:473, 'parent_id'] = 'Diseases of the nervous system'
disease_codes.loc[474:520, 'parent_id'] = 'Diseases of the eye and adnexa'
disease_codes.loc[521:544, 'parent_id'] = 'Diseases of the ear and mastoid process'
disease_codes.loc[545:621, 'parent_id'] = 'Diseases of the circulatory system'
disease_codes.loc[622:685, 'parent_id'] = 'Diseases of the respiratory system'
disease_codes.loc[686:757, 'parent_id'] = 'Diseases of the digestive system'
disease_codes.loc[758:829, 'parent_id'] = 'Diseases of the skin and subcutaneous tissue'
disease_codes.loc[830:908, 'parent_id'] = 'Diseases of the musculoskeletal system and connective tissue'
disease_codes.loc[909:990, 'parent_id'] = 'Diseases of the genitourinary system'
disease_codes.loc[991:1066, 'parent_id'] = 'Pregnancy, childbirth and the puerperium'
disease_codes.loc[1067:1125, 'parent_id'] = 'Certain conditions originating in the perinatal period'
disease_codes.loc[1126:1212, 'parent_id'] = 'Congenital malformations, deformations and chromosomal abnormalities'
disease_codes.loc[1212:, 'parent_id'] = 'Symptoms, signs and abnormal clinical and laboratory findings'

# E47 is missing from UK data 
disease_codes_a = disease_codes.loc[:291, :]
disease_codes_a.loc[292, :] = ['E47', 'E47 something', 'Endocrine, nutritional and metabolic diseases']
disease_codes_b = disease_codes.loc[292:, :]

disease_codes = np.concatenate((np.asarray(disease_codes_a), np.asarray(disease_codes_b)))

np.save(ROOT_DIR + 'projects/CancerRisk/data/prep/disease_codes.npy', disease_codes)

chapter_starts = [0, 173, 221, 255, 329, 407, 475, 522, 546, 623, 687, 759, 831, 910, 992, 1068, 1127, 1214]
chapter_ends = [173, 221, 255, 329, 407, 475, 522, 546, 623, 687, 759, 831, 910, 992, 1068, 1127, 1214, 1305]
chapter_names = ['Certain infectious and parasitic diseases', 
                'Neoplasms', 
                'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism',
                'Endocrine, nutritional and metabolic diseases',
                'Mental and behavioural disorders',
                'Diseases of the nervous system',
                'Diseases of the eye and adnexa',
                'Diseases of the ear and mastoid process',
                'Diseases of the circulatory system',
                'Diseases of the respiratory system',
                'Diseases of the digestive system',
                'Diseases of the skin and subcutaneous tissue',
                'Diseases of the musculoskeletal system and connective tissue',
                'Diseases of the genitourinary system',
                'Pregnancy, childbirth and the puerperium',
                'Certain conditions originating in the perinatal period',
                'Congenital malformations, deformations and chromosomal abnormalities',
                'Symptoms, signs and abnormal clinical and laboratory findings']


with open(ROOT_DIR + 'projects/CancerRisk/data/prep/chapters.pickle', 'wb') as handle:
    pickle.dump({'chapter_starts':chapter_starts, 
                'chapter_ends': chapter_ends, 
                'chapter_names': chapter_names}, handle, protocol=pickle.HIGHEST_PROTOCOL)