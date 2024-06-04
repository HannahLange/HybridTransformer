import torch
import numpy as np
import random
from localenergy import *
import os
import ot
from ot.lp import wasserstein_1d


def cost_fct_samples(model, device, params, bounds_x, bounds_y, antisym, num_samples=None, exp_samples=None):
    """
    Cost function for data-driven pretraining based on Kullback-Leibler divergence.
    """
    cost = 0
    if exp_samples != None:
        for basis in exp_samples.keys():
            if exp_samples[basis] != None:
                # select num_samples samples
                if num_samples < exp_samples[basis].shape[0]:
                    exp_samples_batch = exp_samples[basis].to(device)[torch.randperm(exp_samples[basis].shape[0])][:num_samples]
                else:
                    exp_samples_batch = exp_samples[basis].to(device)
                # calculate the true distribution q
                true_samples, true_freq = torch.unique(exp_samples_batch, dim=0, return_counts=True)
                p_true = true_freq/exp_samples_batch.shape[0]
                # calculate the NQS distribution log(p)
                p_TQS, phases = model.compute_psi(true_samples, check_duplicate=False)
                if basis!= "Z":
                    raise NotImplementedError
                cost -= torch.sum(p_true*p_TQS) #torch.nn.KLDivLoss()(p_TQS, p_true) 
        return cost
    else:
        samples, weights = model.sample(num_samples)
        Eloc, log_probs, phases = get_Eloc(params,samples, model, bounds_x, bounds_y, antisym)
        log_psi = (0.5*log_probs+1j*phases)
        eloc_sum = (weights*Eloc).sum(axis=0)
        e_loc_corr = Eloc - eloc_sum
        cost += 2 * torch.real(torch.conj(weights*log_psi) * (e_loc_corr.detach().to(torch.complex128))).sum()
        return Eloc, cost, samples, weights, log_probs, phases

def cost_fct_samples_ot(model, device, params, bounds_x, bounds_y, antisym, num_samples=None, exp_samples=None, p=1):
    """
    Cost function for data-driven pretraining based on Wasserstein distance and optimal transport (ot).
    """
    cost = 0
    if exp_samples != None:
        for basis in exp_samples.keys():
            if exp_samples[basis] != None:
                # select num_samples samples
                if num_samples < exp_samples[basis].shape[0]:
                    exp_samples_batch = exp_samples[basis].to(device)[torch.randperm(exp_samples[basis].shape[0])][:num_samples]
                else:
                    exp_samples_batch = exp_samples[basis].to(device)
                # calculate the true distribution q
                true_samples, true_freq = torch.unique(exp_samples_batch, dim=0, return_counts=True)
                p_true = true_freq/exp_samples_batch.shape[0]
                # TQS samples and probabilities
                TQS_samples, weights = model.sample(num_samples)
                samples = torch.cat([true_samples, TQS_samples], axis=0)
                # diagonal part of energy as ground metric
                Eloc, log_probs, phases = get_Eloc(params, samples, model, bounds_x, bounds_y, antisym, diag=True)
                p_TQS = torch.exp(log_probs[true_samples.shape[0]:])/(torch.exp(log_probs[true_samples.shape[0]:]).sum())
                dist_true = torch.real(Eloc[:true_samples.shape[0]])
                dist_TQS = torch.real(Eloc[true_samples.shape[0]:])
                if basis != "Z":
                    raise NotImplementedError
                cost += wasserstein_1d(dist_TQS, dist_true, p_TQS, p_true, p=p) / true_samples.shape[1] * 1/params["delta"]
        return cost
    else:
        samples, weights = model.sample(num_samples)
        Eloc, log_probs, phases = get_Eloc(params,samples, model, bounds_x, bounds_y, antisym)
        log_psi = (0.5*log_probs+1j*phases)
        eloc_sum = (weights*Eloc).sum(axis=0)
        e_loc_corr = Eloc - eloc_sum
        cost += 2 * torch.real(torch.conj(weights*log_psi) * (e_loc_corr.detach().to(torch.complex128))).sum()
        return Eloc, cost, samples, weights, log_probs, phases


def cost_fct_corr(model, device, params, bounds_x, bounds_y, antisym, num_samples=None, exp_corrs=None, true_mag=None, experimental_data=True):
    """
    Cost function for data-driven pretraining based on correlation maps.
    """
    cost = 0
    if exp_corrs != None:
        samples, weights = model.sample(num_samples)
        for basis in exp_corrs.keys():
            if true_mag[basis] != None and basis=="Z":
                m = get_mag(samples, weights, basis, model)
                cost += torch.nn.MSELoss(reduction="mean")(m, true_mag[basis].to(torch.float32))
                print("mag",basis, m)
            corr = get_s_corr(samples, weights, basis, model)
            print("corr",basis, corr)
            if exp_corrs[basis] != None:
                cost_corr = 0
                i = 0
                for x in range(samples.shape[1]):
                    for y in range(samples.shape[2]):
                        if x!=int(samples.shape[1]/2) or y!=int(samples.shape[2]/2):
                             d = np.sqrt((x-int(samples.shape[1]/2))**2+(y-int(samples.shape[2]/2))**2)
                             signs = (torch.sign(corr[x,y])==torch.sign(exp_corrs[basis][x,y]))
                             if signs:
                                 if (torch.abs(corr[x,y]) < torch.abs(exp_corrs[basis][x,y])) and experimental_data: #larger penalty if values smaller than the target
                                     factor = 100
                                 elif torch.abs(corr[x,y]) < torch.abs(exp_corrs[basis][x,y]):
                                     factor = 10
                                 else:
                                     factor = 1
                             else:
                                 if experimental_data: factor = 100
                                 else: factor = 10
                             cost_corr += factor * (corr[x,y]-exp_corrs[basis][x,y])**2 / (1-torch.abs(exp_corrs[basis][x,y]))**4
                             i += 1
                cost += cost_corr/i
        print("Corr", cost)
        return cost
    else:
        samples, weights = model.sample(num_samples)
        Eloc, log_probs, phases = get_Eloc(params,samples, model, bounds_x, bounds_y, antisym)
        log_psi = (0.5*log_probs+1j*phases)
        eloc_sum = (weights*Eloc).sum(axis=0)
        e_loc_corr = Eloc - eloc_sum
        cost += 2 * torch.real(torch.conj(weights*log_psi) * (e_loc_corr.detach().to(torch.complex128))).sum()
        return Eloc, cost, samples, weights, log_probs, phases


@torch.no_grad()
def get_mag(samples,weights,basis, model):
    """
    Calculates the local magnetization of samples in a certain basis.
    """
    if basis == "Z":
        log_probs, _ = model.compute_psi(samples)
        probs = torch.exp(log_probs)/torch.exp(log_probs).sum() 
        m = samples.clone()
        m[samples==0] = -1
        mag = torch.zeros((model.system_size[0],model.system_size[0]), device=model.device, dtype=torch.float32)
        for x in range(samples.shape[1]):
            for y in range(samples.shape[2]):
                # no gradients!
                mag[x,y] = torch.sum(weights*m[:,x,y], axis=0)
    else:
        mag = torch.zeros((model.system_size[0],model.system_size[0]), device=model.device, dtype=torch.float32)
        new_samples = [samples.detach()]
        for x in range(samples.shape[1]):
            for y in range(samples.shape[2]):
                batch, Nx, Ny = samples.shape
                samples_x = samples.clone()
                # flip spin
                samples_x[:,x,y] = 1-samples_x[:,x,y] 
                new_samples.append(samples_x)
        new_samples = torch.cat(new_samples, axis=0)
        log_probs, phases = model.compute_psi(new_samples)
        new_samples = new_samples.reshape((-1,batch,samples.shape[1], samples.shape[2]))
        log_probs = log_probs.reshape((-1,batch))
        phases = phases.reshape((-1,batch))
        # calculate \sum_i sigma^x_i
        i = 1
        for x in range(samples.shape[1]):
            for y in range(samples.shape[2]):
                m = torch.exp((0.5*log_probs[i]+1j*phases[i])-(0.5*log_probs[0]+1j*phases[0]))
                probs = torch.exp(log_probs[0])/torch.exp(log_probs[0]).sum() 
                mag[x,y] = torch.sum(weights*m, axis=0)
                i+=1
    return mag

def get_s_corr(samples, weights, basis, model):
    """
    Calculates the spin-spin correlations of samples in a certain basis.
    """
    batch, Nx, Ny = samples.shape
    mid = [[int(Nx/2), int(Ny/2)]]
    if Nx%2==0 and Ny%2==0:
        mid.append([int(Nx/2)-1, int(Ny/2)-1])
    if Nx%2==0:
        mid.append([int(Nx/2)-1, int(Ny/2)])
    if Ny%2==0:
        mid.append([int(Nx/2), int(Ny/2)-1])
    r = torch.randint(0, len(mid), (batch,)) 
    mids0 = torch.stack([torch.tensor(mid[idx][0], device=samples.device) for idx in r], axis=0)
    mids1 = torch.stack([torch.tensor(mid[idx][1], device=samples.device) for idx in r], axis=0)
    corr = torch.zeros((Nx+1,Ny+1), device=model.device, dtype=torch.float64)
    if basis == "Z":
        log_probs, phases = model.compute_psi(samples)
        probs = torch.exp(log_probs)/torch.exp(log_probs).sum() 
        samples_local = samples.clone()
        samples_local[samples==0] = -1
        for x in range(-int(Nx/2),Nx-int(Nx/2)+1):
            for y in range(-int(Ny/2),Ny-int(Ny/2)+1):
                corr[(x+int(Nx/2)),(y+int(Ny/2))] = torch.sum(weights*samples_local[torch.arange(batch),(mids0+x)%Nx,(mids1+y)%Ny] * samples_local[torch.arange(batch),mids0,mids1], axis=0) 
    else:
        new_samples = [samples]
        i = 1
        for x in range(-int(Nx/2),Nx-int(Nx/2)+1):
            for y in range(-int(Ny/2),Ny-int(Ny/2)+1):
                samples_x = samples.clone()
                # flip both spins
                samples_x[torch.arange(batch),(mids0+x)%Nx,(mids1+y)%Ny] = 1-samples_x[torch.arange(batch),(mids0+x)%Nx,(mids1+y)%Ny] 
                samples_x[torch.arange(batch),mids0,mids1] = 1-samples_x[torch.arange(batch),mids0,mids1] 
                new_samples.append(samples_x)
                i += 1
        new_samples = torch.cat(new_samples, axis=0)
        log_probs, phases = model.compute_psi(new_samples)
        new_samples = new_samples.reshape((i,batch,samples.shape[1], samples.shape[2]))
        log_probs = log_probs.reshape((i,batch))
        phases = phases.reshape((i,batch))
        #calculate sigma^x_i*sigma^x_j
        i = 1
        for x in range(Nx+1):
            for y in range(Ny+1):
                c = torch.exp((0.5*log_probs[i]+1j*phases[i])-(0.5*log_probs[0]+1j*phases[0]))
                probs = torch.exp(log_probs[0])/torch.exp(log_probs[0]).sum() 
                corr[x,y] = torch.sum(probs * c, axis=0)
                i += 1
    return corr



