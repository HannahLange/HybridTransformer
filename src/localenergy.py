import torch
import numpy as np
import torch.nn as nn

def XY_MatrixElements(Jp, delta, samples_, boundx, boundy, device, neighbors=[[1,0],[0,1],[1,1],[1,-1],[2,0],[0,2],[2,1],[1,2],[2,-1],[1,-2],[2,2],[2,-2]], dtype=torch.float64, diag=False):
    """ 
    Calculate the local energies of 2D t-J model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - samples: (num_samples, N)
    - Jp: float
    - delta: float
    - boundx, boundy: boundary conditions
    - neighbors: [[1,0],[0,1],[1,1],[1,-1],[2,0],[0,2],[2,1],[1,2],[2,-1],[1,-2],[2,2],[2,-2]]
    """


    Nx         = samples_.size()[1]
    Ny         = samples_.size()[2]
    numsamples = samples_.size()[0]
    samples = samples_.detach().clone()

    # ---------- delta term ---------------
    diag_matrixelements = torch.zeros((numsamples), dtype=dtype, device=device) 
    for i in range(Nx):
        for j in range(Ny):
             if ((i%2==1 and j%2==0) or (i%2==0 and j%2==1)): #sublattice B
                 valuesT = samples[:,i,j].clone()
                 valuesT[valuesT==1]   = -1 #potential for up spins
                 valuesT[valuesT==0]  = 0  #no potential for down spins
                 diag_matrixelements   += valuesT.reshape((numsamples))*delta

    # ---------- vdW term ---------------
    for i in range(Nx): 
        for j in range(Ny):
            for nx,ny in neighbors:
                if boundx=="open": condition1 = ((i+nx)>=0 and (i+nx)<Nx)
                else: condition1 = True
                if boundy=="open": condition2 = ((j+ny)>=0 and (j+ny)<Nx)
                else: condition2 = True
                if condition1 and condition2:
                    values = samples[:,i,j] + samples[:,(i+nx)%Nx,(j+ny)%Ny]
                    valuesT = values.clone().to(torch.float32)
                    valuesT[values==2]   = -0.008 #If both spins are up
                    valuesT[values==0]  = 0.037 #If both spins are down
                    valuesT[values==1]   = 0.0007 #If they are opposite 
                    n = np.sqrt(nx**2+ny**2)
                    diag_matrixelements += (valuesT.reshape((numsamples))/(n**6))
    offd_matrixelements = []
    xprime = []
    if not diag:
    # ---------- S_i+* S_j- term ---------------
        #off-diagonal elements from the S+S- terms
        num = 0

        for i in range(Nx): 
            for j in range(Ny):
                for nx,ny in neighbors:
                    if boundx=="open": condition1 = ((i+nx)>=0 and (i+nx)<Nx)
                    else: condition1 = True
                    if boundy=="open": condition2 = ((j+ny)>=0 and (j+ny)<Nx)
                    else: condition2 = True
                    if condition1 and condition2:
                        new_samples = samples.clone()
                        values = samples[:,i,j] + samples[:,(i+nx)%Nx,(j+ny)%Ny]
                        valuesT = values.clone()
                        new_samples[:,(i+nx)%Nx,(j+ny)%Ny] = samples[:,i,j]
                        new_samples[:,i,j] = samples[:,(i+nx)%Nx, (j+ny)%Ny]
                        valuesT[values==2]   = 0 #If both spins are up
                        valuesT[values==0]  = 0 #If both spins are down
                        valuesT[values==1]   = 1 #If they are opposite 
                        n = np.sqrt(nx**2+ny**2)
                        offd_matrixelements.append(valuesT.reshape((numsamples))*Jp/(n**3))
                        xprime.append(new_samples)
                        num +=1
        offd_matrixelements = torch.stack(offd_matrixelements, axis=-1)
        xprime = torch.stack(xprime, axis=0)    
        x_clone = xprime.clone()
    return diag_matrixelements, offd_matrixelements, xprime



def get_Eloc(parameters, samples, model, boundaries_x, boundaries_y, antisym, diag=False):
    """ 
    Calculate the local energies of 2D t-J model given a set of set of samples.
    Returns: The local energies that correspond to the input samples.
    Inputs:
    - parameters: dict of parameters of the model
    - samples: (num_samples, N)
    - model: NN model
    - boundaries: str, open or periodic
    """
    Jp = parameters["Jp"]
    delta = parameters["delta"]
    device = samples.device

    Nx         = samples.size()[1]
    Ny         = samples.size()[2]
    numsamples = samples.size()[0]

    #matrix elements
    diag_me, offd_me, new_samples = XY_MatrixElements(Jp, delta,samples, boundaries_x, boundaries_y, device, diag=diag)
    # diagonal elements
    Eloc = diag_me.to(torch.complex64)
    if not diag:
        length = new_samples.size()[0]
        offd_me = offd_me.to(torch.complex64)
        # pass all samples together through the network
        queue_samples = torch.zeros(((length+1), numsamples, Nx, Ny), dtype=torch.int32, device=device)  
        queue_samples[0] = samples
        queue_samples[1:length+1] = new_samples
        queue_samples_reshaped = torch.reshape(queue_samples, [(length+1)*numsamples, Nx, Ny])
        log_probs, phases = model.compute_psi(queue_samples_reshaped)
        log_probs_reshaped = torch.reshape(log_probs, (queue_samples.size()[0],numsamples))
        phases_reshaped = torch.reshape(phases, (queue_samples.size()[0],numsamples))

        for i in range(1,(length+1)):
            log_ampl = 0.5*(log_probs_reshaped[i,:]-log_probs_reshaped[0,:])
            phase    = (phases_reshaped[i,:]-phases_reshaped[0,:]).to(log_ampl.dtype)
            Eloc += offd_me[:,i-1]*(torch.exp(torch.complex(log_ampl, phase)))
    else:
        log_probs, phases = model.compute_psi(samples)
        log_probs_reshaped = [log_probs]
        phases_reshaped = [phases]
    return Eloc, log_probs_reshaped[0], phases_reshaped[0]





def get_symmetric_samples(original_samples, symmetry, device):
    samples = original_samples.clone()
    if len(samples.size()) == 3:
        symmetric_samples = torch.zeros((samples.size()[0],samples.size()[1], samples.size()[2]), dtype = torch.int32).to(device)  
        l = int(samples.size()[0])
        symmetric_samples[:l]    = torch.rot90(samples, 2, [1,2])
    else:
        symmetric_samples = torch.zeros((samples.size()[0], samples.size()[1]*3,samples.size()[2], samples.size()[3]), dtype = torch.int32).to(device)  
        l = int(samples.size()[1])
        symmetric_samples[:,:l]    = torch.rot90(samples, 2, [2,3])
    return symmetric_samples

