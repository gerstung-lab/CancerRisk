'''

Main file for the Probabilistic Cox regression
- Cox Partial likelihood
- VI model specification

'''

import tqdm 

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist

import numpy as np
from multiprocessing import Pool

dtype = torch.FloatTensor

np.random.seed(11)
torch.random.manual_seed(12)
pyro.set_rng_seed(13)

# Distributions
# -----------------------------------------------------------------------------------------------------------------------------

class CoxPartialLikelihood(dist.TorchDistribution):
    support = constraints.real
    has_rsample = False

    def __init__(self, pred, sampling_proportion, dtype=dtype):
        self.pred = pred
        self.dtype = dtype
        self.sampling_proportion = sampling_proportion
        super(dist.TorchDistribution, self).__init__()

    def sample(self, sample_shape=torch.Size()):
        return torch.tensor(1.)

    def log_prob(self, surv):
        censor_ratio = torch.tensor([self.sampling_proportion[0]/self.sampling_proportion[1]]).type(self.dtype)
        uncensored_ratio = torch.tensor([self.sampling_proportion[2]/self.sampling_proportion[3]]).type(self.dtype)

        # random tie breaking
        surv[surv[:, -1] == 1, 1] = surv[surv[:, -1] == 1, 1] - torch.normal(0.00001, 0.000001, (torch.sum(surv[:, -1] == 1),)).type(self.dtype)
        event_times = surv[surv[:, -1] == 1, 1][:, None]
        risk_set = ((surv[:, 1] >= event_times) * (surv[:, 0] < event_times)).type(self.dtype)
        aa = torch.sum(self.pred[surv[:, -1] == 1]) *  uncensored_ratio
        bb = torch.sum(torch.log(torch.mm(risk_set, torch.exp(self.pred)) * censor_ratio)) *  uncensored_ratio
        return(aa-bb)


# Models
# -----------------------------------------------------------------------------------------------------------------------------

class PCox():
    def __init__(self, predictor=None, guide=None, levels=None, optimizer=None, scheduler=None, loss=None, sampling_proportion=None, dtype=dtype):
        self.predictor = predictor
        self.guide = guide
        self.levels = levels
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.sampling_proportion = sampling_proportion
        self.dtype = dtype
        super().__init__()

    def model(self, data):
        # prior
        pred = self.predictor(data, level=self.levels)
        
        # stoachstic update -likelihood adjustment
        ll = self.levels

        self.sampling_proportion[ll][0] = torch.tensor([self.sampling_proportion[ll][0]])
        self.sampling_proportion[ll][1] = torch.tensor([self.sampling_proportion[ll][1]])
        self.sampling_proportion[ll][2] = torch.tensor([self.sampling_proportion[ll][2]])
        self.sampling_proportion[ll][3] = torch.sum(data[0][:, -1])

        pyro.sample("obs_", CoxPartialLikelihood(pred=pred, sampling_proportion=self.sampling_proportion[ll], dtype=self.dtype), obs=data[0])

    def make_guide(self, rank):
        if self.guide:
            self.guide = self.guide
        else:
            if rank:
                self.guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(self.model, rank=rank)
            else:
                self.guide = pyro.infer.autoguide.AutoMultivariateNormal(self.model)

    def make_optimizer(self, eta):
        if self.optimizer:
            pass
        else:
            self.optimizer = torch.optim.SGD
            self.scheduler = pyro.optim.ExponentialLR({'optimizer': self.optimizer, 'optim_args': {'lr': eta}, 'gamma': 0.80})


    def make_loss(self, num_particles):
        if self.loss:
            pass
        else:
            self.loss = pyro.infer.Trace_ELBO(num_particles=num_particles, strict_enumeration_warning=False)

    def return_guide(self):
        return(self.guide)

    def return_model(self):
        return(self.model)

    def initialize(self, seed=874, num_particles=1, eta=0.001, rank=None):
        pyro.set_rng_seed(seed)
        self.make_guide(rank)
        self.make_optimizer(eta)
        self.make_loss(num_particles)
        self.svi = pyro.infer.SVI(model=self.model, guide=self.guide, optim=self.scheduler, loss=self.loss)

    def infer(self, data):
        ll = self.svi.step(data)

        return(ll)
    
    def decay(self):
        self.scheduler.step()


# Metrics
def concordance(surv, predictions, return_pairs=False):
    with torch.no_grad():
        surv = torch.from_numpy(surv)
        predictions = torch.from_numpy(predictions)
        # small + offset for censored data
        surv[surv[:, -1]==1, 1] = surv[surv[:, -1]==1, 1] - 0.0000001
        event_times = surv[surv[:, -1]==1, 1]
        event_hazard = predictions[surv[:, -1]==1]
        concordant = 0
        disconcordant = 0
        tx = 0
        ty = 0
        txy = 0
        for ii in (range(event_times.shape[0])):
            risk_set = (surv[:, 0] < event_times[ii]) * (event_times[ii] < surv[:, 1])
            txy += torch.sum((event_times[ii] == surv[:, 1]) *(event_hazard[ii] == predictions))-1
            ty += torch.sum(event_times[ii] == surv[:, 1])-1
            tx += torch.sum(event_hazard[ii] == predictions[risk_set])
            concordant += torch.sum(predictions[risk_set] < event_hazard[ii])
            disconcordant += torch.sum(predictions[risk_set] > event_hazard[ii])

        if return_pairs:
            return(concordant, disconcordant, txy, tx, ty)
        else:
            return(((concordant-disconcordant) / (concordant+disconcordant+tx)+1)/2)

        
def KM(times, events, t1=None, tmax=3700):
    '''
    '''
    #tmax = np.max(times)
    
    
    idx = np.argsort(times)
    times = times[idx]
    events = events[idx]
    events[times>tmax]=0
    times = np.minimum(times, tmax)


    helpvar = np.arange(times.shape[0], 0, -1)
    ll = []
    if np.any(t1!=None):
        for tt in t1:
            try:
                idx_m = np.max(np.where(times <= tt)[0])
                ll.extend([[events[:idx_m].sum(), helpvar[idx_m]]])
            except:
                ll.extend([[0, helpvar[0]]])

    idx_cases = events == 1
    
    if np.sum(idx_cases) > 0:
        times = times[idx_cases]
        events = events[idx_cases]
        helpvar = helpvar[idx_cases]

        events = np.asarray([np.sum(events[jj == times], axis=0) for jj in np.unique(times)])
        helpvar = np.asarray([np.max(helpvar[jj == times], axis=0) for jj in np.unique(times)])
        times = np.unique(times)
        km = np.cumprod((1 - events/helpvar))
        km_helpvar = np.cumsum(events/(helpvar*(helpvar-events)))

        times = np.concatenate(([0], times, [tmax]))
        km = np.concatenate(([0.9999999],  km, [km[-1]]))
        km_helpvar = np.concatenate(([0],  km_helpvar, [km_helpvar[-1]]))

        # log-log Variance estimation
        V = 1/np.log(km)**2 * km_helpvar
        lower = np.exp(-np.exp(np.log(-np.log(km)) + 1.96*np.sqrt(V)))
        upper = np.exp(-np.exp(np.log(-np.log(km)) - 1.96*np.sqrt(V)))
    else:

          
        times = np.asarray([0, tmax])
        km = np.asarray([1, 1])
        lower = np.asarray([1, 1])
        upper = np.asarray([1, 1])
                
    return(times, km, [lower, upper], np.asarray(ll))



def Breslow(times, pred):
    times[times[:, -1]==1, 1] = times[times[:, -1]==1, 1] - 0.0000001
    event_times = times[times[:, -1] ==1, 1]
    event_times = event_times[np.argsort(event_times)]
    a0 = [0]
    for ii in tqdm.tqdm(range(event_times.shape[0])):
        risk_set = (times[:, 0] < event_times[ii]) * (event_times[ii] <= times[:, 1])
        a0.append(1/np.sum(np.exp(pred[risk_set])))
    return(event_times, np.asarray(a0[1:]))