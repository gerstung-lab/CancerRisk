## Modules
#=======================================================================================================================
import sys
import os 
import tqdm
import h5py
import subprocess
import time
import numpy as np 

## Custom Functions
#=======================================================================================================================
def KM(times, events, t1=None, tmax=None):
    '''
    updated KM - added log-log variance estimator + events/at risk counter
    '''
    
    if tmax==None:
        tmax = np.max(times)
        
    idx = np.argsort(times)
    times = times[idx]
    events = events[idx]
    events[times>tmax]=0
    times= np.minimum(times, tmax)
    
    helpvar = np.arange(times.shape[0], 0, -1)
    ll=[]
    if np.any(t1!=None):
        for tt in t1:
            try:
                idx_m = np.max(np.where(times <= tt)[0])
                ll.extend([[events[:idx_m].sum(), helpvar[idx_m]]])
            except:
                ll.extend([[0, helpvar[0]]])
                
    idx_cases = events == 1
    
    if np.sum(idx_cases)>0:
        
        times = times[idx_cases]
        events = events[idx_cases]
        helpvar = helpvar[idx_cases]

        events = np.asarray([np.sum(events[jj == times], axis=0) for jj in np.unique(times)])
        helpvar = np.asarray([np.max(helpvar[jj == times], axis=0) for jj in np.unique(times)])
        times = np.unique(times)
        km = np.cumprod((1 - events/helpvar))
        km_helpvar = np.cumsum(events/(helpvar*(helpvar-events)))
        
        
        times = np.concatenate(([0], times, [tmax]))
        km = np.concatenate(([0.9999999999], km, [km[-1]]))
        km_helpvar = np.concatenate(([0], km_helpvar, [km_helpvar[-1]]))
        
        # log-log variance estimator
        V = 1/np.log(km)**2 * km_helpvar
        lower = np.exp(-np.exp(np.log(-np.log(km)) + 1.96*np.sqrt(V)))
        upper = np.exp(-np.exp(np.log(-np.log(km)) - 1.96*np.sqrt(V)))
    else:
        times = np.asarray([0, tmax])
        km = np.asarray([1, 1])
        lower = np.asarray([1, 1])
        upper = np.asarray([1, 1])

    return(times, km, [lower, upper], np.asarray(ll))

class CIF():
    def __init__(self, cc, tt0, tt_range, A0, pred, sex, full=False, **kwds):
        self.cc = cc
        self.tt0 = tt0
        self.tt_range = tt_range
        self.A0 = A0 
        self.pred = pred
        self.sex = sex
        self.full = full 
        self.sexnames = ['female', 'male']
        
    def __call__(self, ii):
        A0net = np.sum((self.A0[:, self.sex[ii], self.tt0[ii]:self.tt0[ii]+self.tt_range]*np.exp(self.pred[ii, :, None])), axis=0)
        S0net = np.exp(-np.cumsum(A0net))
        
        if self.full:
            pest = np.cumsum(self.A0[:20, self.sex[ii], self.tt0[ii]:self.tt0[ii]+self.tt_range]*np.exp(self.pred[ii, :20, None]) * S0net[None, :], axis=1)[:, -1].sum()
            
        else:
            pest = np.cumsum(self.A0[self.cc, self.sex[ii], self.tt0[ii]:self.tt0[ii]+self.tt_range] * np.exp(self.pred[ii, self.cc, None]) * S0net)[-1]
        return([pest])
    
def metric_table(pr, re):
    # produces contingency table for actual vs predicted binary outcomes.
    # pr == predicted 
    # re == actual
    # contigency table based on https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values - relationship section
    
    
    P = re.sum()#positive
    N = re.shape[0] - P #negative
    T = P + N

    # predicted 
    PP = pr.sum()#positive
    PN = (~pr).sum()#negative

    # contingency table 
    TP = np.logical_and(re, pr).sum()
    FN = np.logical_and(re, ~pr).sum()
    FP = np.logical_and(~re, pr).sum()
    TN = np.logical_and(~re, ~pr).sum()

    conti = np.zeros((4, 4))
    conti[0, 0] = P+N
    conti[1, 0] = P
    conti[2, 0] = N
    conti[3, 0] = P/(P+N)

    conti[0, 1] = PP
    conti[1, 1] = TP
    conti[2, 1] = FP
    conti[3, 1] = TP/PP

    conti[0, 2] = PN
    conti[1, 2] = FN
    conti[2, 2] = TN
    conti[3, 2] = FN/PN

    conti[0, 3] = (TP/P) + (TN/N) -1
    conti[1, 3] = TP/P
    conti[2, 3] = FP/N
    conti[3, 3] = (TP/P)/(FP/N)

    return(conti)

def round_(x, r):
    x = str(np.round(x, r))
    if len(x) < r+2:
        x = x + '0' * (r+2-len(x))
    return(x)


def quantile(a, b):
    try:
        rr = np.quantile(a, b)
    except:
        rr = np.zeros((len(b),))
    return(rr)