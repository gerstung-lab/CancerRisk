# %%
import os
ROOT_DIR = '/users/projects/cancer_risk/main/'

events = ['oesophagus', 'stomach', 'colorectal', 'liver', 'pancreas', 'lung', 'melanoma', 'breast', 
                'cervix_uteri', 'corpus_uteri', 'ovary', 'prostate', 'testis', 'kidney', 'bladder', 'brain',
                'thyroid', 'non_hodgkin_lymphoma', 'multiple_myeloma', 'AML', 'other', 'death']

for cc in range(len(events)):
    os.mkdir(ROOT_DIR + 'output/' + events[cc])
    os.mkdir(ROOT_DIR + 'output/' + events[cc] + '/data')
    os.mkdir(ROOT_DIR + 'output/' + events[cc] + '/tables')
    os.mkdir(ROOT_DIR + 'output/' + events[cc] + '/figures')

os.mkdir(ROOT_DIR + 'output/' + 'main')
os.mkdir(ROOT_DIR + 'output/' + 'main' + '/data')   
os.mkdir(ROOT_DIR + 'output/' + 'main' + '/tables')
os.mkdir(ROOT_DIR + 'output/' + 'main' + '/figures')



# %%



