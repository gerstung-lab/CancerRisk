#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import os
import datetime
import h5py
import glob
import pickle
import pandas as pd
import numpy as np 
import multiprocessing as mp
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib as mpl

from multiprocessing import Pool

np.random.seed(seed=83457)

d_data = '/home/people/alexwolf/data/'

# refernce files
ref_DB = np.load(d_data + 'DB/DB/raw/ref.npy')
ref_cancer = pickle.load(open(d_data + 'DB/cancer/_ref.pickle', "rb"))


# In[ ]:


# varibale passed from job submission
counter = int(sys.argv[1])


# In[3]:


# custom functions
def _find(idx, ref):
    k = 9999
    for key, value in ref.items():
        if idx in value: 
            k = key
            break 
    return(k)

def _or(x, y):
    return np.logical_or(x, y)

def _and(x, y):
    return np.logical_and(x, y)

def _minimum(x, y):
    return np.minimum(x, y)

def _maximum(x, y):
    return np.maximum(x, y)

def remove_D_coding(x):
    ll = []
    for i in x:
        ll.append(i.replace('DC', 'C').replace('DD', 'D').replace('DZ', 'Z'))
    return(np.asarray(ll))

def _adjust_dr(x):
    try:
        helpvar = []
        for jj in x[:, 1:].tolist()[0]:
            helpvar.extend(jj.replace(' ', '').split(','))
        helpvar = np.asarray(helpvar)[None, :]
        helpvar = np.unique(helpvar)
        helpvar = helpvar[helpvar != 'nan']
        helpvar = helpvar[helpvar != '']
        x = np.concatenate((np.asarray(x[0, 0])[None, None], helpvar[None, :]), axis=1)[0]
        return(x)
    except:
        return(x)

def min_date(dnpr, cr, dr, cancer):
    try:
        d_dnpr = np.datetime64(dnpr[dnpr[:, 1] == cancer, 0][0])
    except:
        d_dnpr = np.datetime64('2020-01-01')
    try:
        d_cr = np.datetime64(cr[cr[:, -1] == cancer, 0][0])
    except:
        d_cr = np.datetime64('2020-01-01')
    try:
        if cancer in dr[1:].tolist():
            d_dr = np.datetime64(dr[0])
        else: 
            d_dr = np.datetime64('2020-01-01')
    except:
        d_dr = np.datetime64('2020-01-01')
        
    return(_minimum(_minimum(d_dnpr, d_cr), d_dr).astype(str))

def _combine(dnpr, cr, dr):
    cc_ = []
    cc_.extend(dnpr[:, 1])
    cc_.extend(cr[:, -1])
    cc_.extend(dr[1:])
    cc = []
    for jj in cc_:
        cc.extend(jj.replace(' ', '').split(','))
    cc = np.unique(cc)

    dd = []
    for cancer in cc:
        dd.append(min_date(dnpr, cr, dr, cancer))
        
    mm = []  
    for c in cc:
        if c in cr[:, -1]:
            mm.append(int(cr[c == cr[:, -1], -2][0]))
        else: 
            mm.append(0)
    return(np.concatenate((np.asarray(cc)[:, None], np.asarray(dd)[:, None], np.asarray(mm)[:, None]), axis=1))

def min_date_full(dnpr, cr, dr, cancer):
    try:
        d_dnpr = np.datetime64(dnpr[dnpr[:, -2] == cancer, 0][0])
    except:
        d_dnpr = np.datetime64('2020-01-01')
    try:
        d_cr = np.datetime64(cr[cr[:, 2] == cancer, 0][0])
    except:
        d_cr = np.datetime64('2020-01-01')
    try:
        if cancer in dr[1:].tolist():
            d_dr = np.datetime64(dr[0])
        else: 
            d_dr = np.datetime64('2020-01-01')
    except:
        d_dr = np.datetime64('2020-01-01')
    return(_minimum(_minimum(d_dnpr, d_cr), d_dr).astype(str))

def _combine_full(dnpr, cr, dr):
    cc_ = []
    cc_.extend(dnpr[:, -2])
    cc_.extend(cr[:, 2])
    cc_.extend(dr[1:])
    cc = []
    for jj in cc_:
        cc.extend(jj.replace(' ', '').split(','))
    cc = np.unique(cc)

    dd = []
    for cancer in cc:
        dd.append(min_date_full(dnpr, cr, dr, cancer))
        
    mm = []  
    for c in cc:
        if c in cr[:, 2]:
            mm.append(int(cr[c == cr[:, 2], -2][0]))
        else: 
            mm.append(0)
    return(np.concatenate((np.asarray(cc)[:, None], np.asarray(dd)[:, None], np.asarray(mm)[:, None]), axis=1))

def _write(idx):
    file_id = _find(idx, ref_cancer)
    if file_id == 9999:
        dnpr = np.zeros((1,7))[[]]
        cr = np.zeros((1,5))[[]]
        dr = np.zeros((1,14))[[]]
        dr_add = np.zeros((1,14))[[]]
        comb = np.zeros((1,3))[[]]
        comb_full = np.zeros((1,3))[[]]
        cancer_idx = np.asarray([0]).astype(int)
        cancer = np.asarray([]).astype('S4')
        cancer_full = np.asarray([]).astype('S6')
        morph = np.asarray([]).astype(int)
        date = np.asarray([]).astype('S10')
        quality = np.asarray([]).astype(int)
    else:
        with h5py.File(d_data + 'DB/cancer/_' + str(file_id) + '.h5', 'r') as f:
            dnpr = f[idx]['DNPR'][:].astype(str)
            idx_dnpr_malignant = [dnpr[kk, 1] in malignant_cancer_codes for kk in np.arange(dnpr.shape[0])]
            if dnpr.shape[0] != 0:
                dnpr[:, -2] = remove_D_coding(dnpr[:, -2])

            cr = f[idx]['CR'][:].astype(str)
            cr[cr == 'nan'] = '0'
            idx_cr_malignant = [cr[kk, -1] in malignant_cancer_codes for kk in np.arange(cr.shape[0])]

            dr = f[idx]['DR'][:, :14].astype(str)
            dr_add = np.concatenate((f[idx]['DR'][:, 0].astype(str)[:, None], f[idx]['DR'][:, 14:].astype(str)), axis=1)
            dr_add[dr == ''] = ''
            
            dr =_adjust_dr(dr)
            dr_add = _adjust_dr(dr_add)

            comb = _combine(dnpr, cr, dr)
            comb_full = _combine_full(dnpr, cr, dr_add)

            # both match
            if _and(dnpr[idx_dnpr_malignant, 1].size != 0, cr[idx_cr_malignant, -1].size != 0):

                # match - *1
                if dnpr[idx_dnpr_malignant, 1][0] == cr[idx_cr_malignant, -1][0]:
                    cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                    cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                    morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                    quality = np.asarray(1).astype(int)[None]

                    # minimum DNPR date with 1 year window for uncertain cancer 
                    if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                        dd_1 = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                    else: 
                        dd_1 = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))

                    # minimum CR date with 1 year window for uncertain cancer 
                    if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                        dd_2 = np.datetime64(cr[idx_cr_malignant, 0][0])
                    else: 
                        dd_2 = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))

                    date = np.minimum(dd_1, dd_2)

                # both - non primary match
                else:
                    #DNPR - validation
                    dnpr_count = dnpr[idx_dnpr_malignant, -1].astype(int)[0]
                    dnpr_dr_entry = bool(_minimum(1, sum(dnpr[idx_dnpr_malignant, 1][0] == dr[kk] for kk in np.arange(1, dr.shape[0]))))
                    dnpr_cr_entry = bool(_minimum(1, sum(dnpr[idx_dnpr_malignant, 1][0] == cr[kk, -1] for kk in np.arange(cr.shape[0]))))

                    if dnpr[:, 0].size >= 2:
                        dnpr_sub = dnpr[dnpr[idx_dnpr_malignant, 0][0].astype('datetime64') < dnpr[:, 0].astype('datetime64'), :]
                        if dnpr_sub.size != 0:
                            dnpr_followup = bool(np.minimum(1, sum(np.asarray([dnpr_sub[kk, 1] in malignant_cancer_treatment_codes for kk in np.arange(dnpr_sub.shape[0])]).astype(int))))
                        else:
                            dnpr_followup = bool(0)
                    else:
                        dnpr_followup = bool(0)

                    dnpr_val = _or(_or(_or(dnpr_followup, dnpr_cr_entry), dnpr_dr_entry), dnpr_count > 1)

                    # CR - validation 
                    cr_dr_entry = bool(_minimum(1, sum(cr[idx_cr_malignant, -1][0] == dr[kk] for kk in np.arange(1, dr.shape[0]))))
                    cr_dnpr_entry = bool(_minimum(1, sum(cr[idx_cr_malignant, -1][0] == dnpr[kk, 1] for kk in np.arange(dnpr.shape[0]))))
                    if dnpr.size != 0 and cr[idx_cr_malignant, 0].size != 0:
                        dnpr_sub = dnpr[cr[idx_cr_malignant, 0][0].astype('datetime64') < dnpr[:, 0].astype('datetime64'), :]
                        if dnpr_sub.size != 0:
                            cr_followup = bool(np.minimum(1, np.sum(np.asarray([dnpr_sub[kk, 1] in malignant_cancer_treatment_codes for kk in np.arange(dnpr_sub.shape[0])]).astype(int))))
                        else:
                            cr_followup = bool(0)
                    else:
                        cr_followup = bool(0)
                    cr_val = _or(_or(cr_followup, cr_dnpr_entry), cr_dr_entry)

                    # DNPR only validation + < 1year - *8
                    if _and(cr_val == False, dnpr_val == True):        
                        cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                        cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                        morph = np.asarray(0).astype(int)[None]
                        quality = np.asarray(8).astype(int)[None]

                        # minimum DNPR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                            dd_1 = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                        else: 
                            dd_1 = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))

                        # minimum CR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                            dd_2 = np.datetime64(cr[idx_cr_malignant, 0][0])
                        else: 
                            dd_2 = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))

                        if (dd_1 - dd_2).astype(float) < -365:
                            date = dd_1
                        else:
                            date = np.minimum(dd_1, dd_2)


                    # CR only validation + < 1year - *9
                    elif _and(cr_val == True, dnpr_val == False):
                        cancer = np.asarray(cr[idx_cr_malignant, -1][0]).astype('S4')[None]
                        cancer_full = np.asarray(cr[idx_cr_malignant, 2][0]).astype('S6')[None]
                        morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                        quality = np.asarray(9).astype(int)[None]

                        # minimum DNPR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                            dd_1 = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                        else: 
                            dd_1 = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))

                        # minimum CR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                            dd_2 = np.datetime64(cr[idx_cr_malignant, 0][0])
                        else: 
                            dd_2 = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))

                        if (dd_2 - dd_1).astype(float) < -365:
                            date = dd_2
                        else:
                            date = np.minimum(dd_1, dd_2)

                    # both + validation
                    elif _and(cr_val == True, dnpr_val == True):
                        # minimum DNPR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                            dd_1 = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                        else: 
                            dd_1 = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))

                        # minimum CR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                            dd_2 = np.datetime64(cr[idx_cr_malignant, 0][0])
                        else: 
                            dd_2 = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))

                        if dd_1 <= dd_2:
                            cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                            cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                            morph = np.asarray(0).astype(int)[None]
                            quality = np.asarray(10).astype(int)[None]
                            date = dd_1
                        else:
                            cancer = np.asarray(cr[idx_cr_malignant, -1][0]).astype('S4')[None]
                            cancer_full = np.asarray(cr[idx_cr_malignant, 2][0]).astype('S6')[None]
                            morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                            quality = np.asarray(10).astype(int)[None]
                            date = dd_2
                    # both but uncertain - 13
                    else:
                        # minimum DNPR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                            dd_1 = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                        else: 
                            dd_1 = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))

                        # minimum CR date with 1 year window for uncertain cancer 
                        if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                            dd_2 = np.datetime64(cr[idx_cr_malignant, 0][0])
                        else: 
                            dd_2 = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))

                        if dd_1 <= dd_2:
                            cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                            cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                            morph = np.asarray(0).astype(int)[None]
                            quality = np.asarray(13).astype(int)[None]
                            date = dd_1
                        else:
                            cancer = np.asarray(cr[idx_cr_malignant, -1][0]).astype('S4')[None]
                            cancer_full = np.asarray(cr[idx_cr_malignant, 2][0]).astype('S6')[None]
                            morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                            quality = np.asarray(13).astype(int)[None]
                            date = dd_2


            # DNPR only 
            elif _and(dnpr[idx_dnpr_malignant, 1].size != 0, cr[idx_cr_malignant, -1].size == 0):
                #DNPR - validation
                dnpr_count = dnpr[idx_dnpr_malignant, -1].astype(int)[0]
                dnpr_dr_entry = bool(_minimum(1, sum(dnpr[idx_dnpr_malignant, 1][0] == dr[kk] for kk in np.arange(1, dr.shape[0]))))
                dnpr_cr_entry = bool(_minimum(1, sum(dnpr[idx_dnpr_malignant, 1][0] == cr[kk, -1] for kk in np.arange(cr.shape[0]))))

                if dnpr[:, 0].size >= 2:
                    dnpr_sub = dnpr[dnpr[idx_dnpr_malignant, 0][0].astype('datetime64') < dnpr[:, 0].astype('datetime64'), :]
                    if dnpr_sub.size != 0:
                        dnpr_followup = bool(np.minimum(1, sum(np.asarray([dnpr_sub[kk, 1] in malignant_cancer_treatment_codes for kk in np.arange(dnpr_sub.shape[0])]).astype(int))))
                    else:
                        dnpr_followup = bool(0)
                else:
                    dnpr_followup = bool(0)

                dnpr_val = _or(_or(_or(dnpr_followup, dnpr_cr_entry), dnpr_dr_entry), dnpr_count > 1)

                if dnpr_dr_entry:
                    cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                    cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                    morph = np.asarray(0).astype(int)[None]
                    quality = np.asarray(2).astype(int)[None]
                    if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))        
                    if cr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(cr[0, 0]))        
                        
                        
                elif dnpr_followup:
                    cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                    cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                    morph = np.asarray(0).astype(int)[None]
                    quality = np.asarray(3).astype(int)[None]
                    if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))        
                    if cr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(cr[0, 0])) 
                        
                elif dnpr_count > 1:
                    cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                    cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                    morph = np.asarray(0).astype(int)[None]
                    quality = np.asarray(4).astype(int)[None]
                    if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))        
                    if cr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(cr[0, 0])) 
                        
                else: 
                    cancer = np.asarray(dnpr[idx_dnpr_malignant, 1][0]).astype('S4')[None]
                    cancer_full = np.asarray(dnpr[idx_dnpr_malignant, -2][0]).astype('S6')[None]
                    morph = np.asarray(0).astype(int)[None]
                    quality = np.asarray(11).astype(int)[None]
                    if ((np.datetime64(dnpr[0, 0]) - np.datetime64(dnpr[idx_dnpr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(dnpr[idx_dnpr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(dnpr[idx_dnpr_malignant, 0][0]))        
                    if cr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(cr[0, 0])) 
                        
            # CR only 
            elif _and(dnpr[idx_dnpr_malignant, 1].size == 0, cr[idx_cr_malignant, -1].size != 0):   
                # CR - validation 
                cr_dr_entry = bool(_minimum(1, sum(cr[idx_cr_malignant, -1][0] == dr[kk] for kk in np.arange(1, dr.shape[0]))))
                cr_dnpr_entry = bool(_minimum(1, sum(cr[idx_cr_malignant, -1][0] == dnpr[kk, 1] for kk in np.arange(dnpr.shape[0]))))
                
                if dnpr.shape[0] !=0:
                    dnpr_sub = dnpr[cr[idx_cr_malignant, 0][0].astype('datetime64') < dnpr[:, 0].astype('datetime64'), :]
                    if dnpr_sub.size != 0:
                        cr_followup = bool(np.minimum(1, np.sum(np.asarray([dnpr_sub[kk, 1] in malignant_cancer_treatment_codes for kk in np.arange(dnpr_sub.shape[0])]).astype(int))))
                    else:
                        cr_followup = bool(0)
                else:
                    cr_followup = bool(0)
                cr_val = _or(_or(cr_followup, cr_dnpr_entry), cr_dr_entry) 

                if cr_dr_entry:
                    cancer = np.asarray(cr[idx_cr_malignant, -1][0]).astype('S4')[None]
                    cancer_full = np.asarray(cr[idx_cr_malignant, 2][0]).astype('S6')[None]
                    morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                    quality = np.asarray(5).astype(int)[None]

                    # minimum CR date with 1 year window for uncertain cancer 
                    if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(cr[idx_cr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))
                    if dnpr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(dnpr[0, 0]))  
                        
                elif cr_followup:
                    cancer = np.asarray(cr[idx_cr_malignant, -1][0]).astype('S4')[None]
                    cancer_full = np.asarray(cr[idx_cr_malignant, 2][0]).astype('S6')[None]
                    morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                    quality = np.asarray(6).astype(int)[None]

                    # minimum CR date with 1 year window for uncertain cancer 
                    if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(cr[idx_cr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))
                    if dnpr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(dnpr[0, 0]))  
                        
                else: 
                    cancer = np.asarray(cr[idx_cr_malignant, -1][0]).astype('S4')[None]
                    cancer_full = np.asarray(cr[idx_cr_malignant, 2][0]).astype('S6')[None]
                    morph = np.asarray(cr[idx_cr_malignant, -2][0]).astype(int)[None]
                    quality = np.asarray(12).astype(int)[None]

                    # minimum CR date with 1 year window for uncertain cancer 
                    if ((np.datetime64(cr[0, 0]) - np.datetime64(cr[idx_cr_malignant, 0][0])).astype(float)) < -365:
                        date = np.datetime64(cr[idx_cr_malignant, 0][0])
                    else: 
                        date = np.minimum(np.datetime64(cr[0, 0]), np.datetime64(cr[idx_cr_malignant, 0][0]))
                    if dnpr.shape[0] != 0:
                        date = _minimum(date, np.datetime64(dnpr[0, 0]))  
                        
            # DR only 
            elif _and(_and(dnpr[idx_dnpr_malignant, 1].size == 0, cr[idx_cr_malignant, -1].size == 0), dr.shape[0] != 0):
                cancer = np.asarray(dr[1]).astype('S4')[None]
                cancer_full = np.asarray(dr_add[1]).astype('S6')[None]
                morph = np.asarray([0]).astype(int)[None]
                quality = np.asarray(7).astype(int)[None]
                date = dr[0]

            # undefined
            else: 
                cancer = np.asarray(['C100']).astype('S4')[None]
                cancer_full = np.asarray(['C100']).astype('S6')[None]
                morph = np.asarray([0]).astype(int)[None]
                quality = np.asarray(14).astype(int)[None]
                if _and(dnpr.size != 0, cr.size != 0):
                    date = np.minimum(np.datetime64(dnpr[0, 0]), np.datetime64(cr[0, 0])) 
                elif dnpr.size != 0:
                    date = dnpr[0, 0]
                else: 
                    date = cr[0, 0]

            date = [date.astype(str).astype('S10')]
            cancer_idx = np.asarray([1]).astype(int)
            cancer = np.asarray(cancer).astype('S4')
            cancer_full = np.asarray(cancer_full).astype('S6')
            morph = np.asarray(morph).astype(int)
            date = np.asarray(date).astype('S10')
            quality = np.asarray(quality).astype(int)
            
            comb[:, 1] = comb[:, 1].astype('datetime64')
            idx_order = np.argsort(comb[:, 1])
            comb = comb[idx_order, :]
            comb = comb.astype('S10')

            comb_full[:, 1] = comb_full[:, 1].astype('datetime64')
            idx_order = np.argsort(comb_full[:, 1])
            comb_full = comb_full[idx_order, :]
            comb_full = comb_full.astype('S10')
            if dr.shape[0] != 0:
                dr = dr[None, :]
                dr_add = dr_add[None, :]
            

    with h5py.File(d_data + 'DB/DB/raw/_' + str(ind), 'a') as f:
        
        try: 
            f[idx]['cancer']['set_full']
        except:
            f[idx].attrs['cancer'] = cancer_idx
            f[idx].create_group('cancer')
            f[idx]['cancer'].attrs['icd10'] = cancer
            f[idx]['cancer'].attrs['icd10_full'] = cancer_full
            f[idx]['cancer'].attrs['date'] = date
            f[idx]['cancer'].attrs['morph'] = morph
            f[idx]['cancer'].attrs['quality'] = quality
            f[idx]['cancer'].create_dataset('DNPR', data=dnpr.astype('S10'), maxshape=(None, 7), compression="lzf")
            f[idx]['cancer'].create_dataset('CR', data=cr.astype('S10'), maxshape=(None, 5), compression="lzf")
            f[idx]['cancer'].create_dataset('DR', data=dr.astype('S10'), maxshape=(None, 14), compression="lzf")
            f[idx]['cancer'].create_dataset('DR_full', data=dr_add.astype('S10'), maxshape=(None, 14), compression="lzf")
            f[idx]['cancer'].create_dataset('set', data=comb.astype('S10'), maxshape=(None, 3), compression="lzf")
            f[idx]['cancer'].create_dataset('set_full', data=comb_full.astype('S10'), maxshape=(None, 3), compression="lzf")


# In[4]:


# defining lists for diagnosisi
# uncertain cancer diagnosis 
uncertain_cancer_codes = ['D' + str(i) for i in np.arange(37, 45)]
uncertain_cancer_codes.extend(['D' + str(i) for i in np.arange(47, 49)])

# bening cancer
benign_cancer_codes = ['D' + str(i) for i in np.arange(10, 37)]
benign_cancer_codes.extend(['D0' + str(i) for i in np.arange(0, 10)])
benign_cancer_codes.extend(['D45', 'D46'])

# malignant cancer
malignant_cancer_codes = ['C' + str(i) for i in np.arange(10, 77)]
malignant_cancer_codes.extend(['C' + str(i) for i in np.arange(80, 100)])
malignant_cancer_codes.extend(['C0' + str(i) for i in np.arange(0, 10)])

# treatment for malignant cancer
malignant_cancer_treatment_codes = ['Z08', 'Z85']

# secondary cancers 
secondary_cancer_codes = ['C77', 'C78', 'C79']

relevant_diagnosis = secondary_cancer_codes + malignant_cancer_treatment_codes + malignant_cancer_codes + uncertain_cancer_codes
cancer_diagnosis = malignant_cancer_codes + secondary_cancer_codes


# In[5]:


for ind in np.arange(692, 701):
    print(ind)
    for idx in ref_DB[ind]:
        _write(idx)


# In[ ]:


print('finsihed')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




