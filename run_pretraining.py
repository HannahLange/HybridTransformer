import sys
sys.path.insert(1, 'src')

# load transformer, helper functions and cost functions
from model import TransformerModel
from helper import *
from symmetry import *
from cost import *

#load general modules
import argparse
import os
import timeit
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                if torch.cuda.is_available()
                                else torch.FloatTensor)

parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   , type=int,  default = 6 ,    help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   , type=int,  default = 6 ,    help="length in y dir")
parser.add_argument("-J"  , "--J"    , type=float,default = 0.77 , help="J (spin interaction)")
parser.add_argument("-d"  , "--delta", type=float,default = 1 ,    help="light shift delta")
parser.add_argument("-hd" , "--hd"   , type=int,  default = 32 ,   help="hidden dimension")
parser.add_argument("-nl" , "--nl"   , type=int,  default = 1 ,    help="number of TQS layers")
parser.add_argument("-nh" , "--nh"   , type=int,  default = 8 ,    help="number of TQS heads")
parser.add_argument("-sym", "--sym"  , type=int,  default = 0 ,    help="spatial symmetries (0:False, 1:True)")


torch.set_default_tensor_type(torch.cuda.FloatTensor
                                if device=="cuda:0"
                                else torch.FloatTensor)


args = parser.parse_args()
Nx     = args.Nx
Ny     = args.Ny
N_tot  = Nx * Ny
antisym = False
params = {"Jp": args.J, "delta": args.delta}
system_sizes = [Nx,Ny]


# training hyperparameters
lr = 1e-5
num_epochs = 1500
num_samples = 256*2

# transformer parameters
patchsize = [2,2]
embedding_size = args.hd
n_head = args.nh
n_hid = embedding_size*5
n_layers = args.nl
dropout = 0
minibatch = 100000
symmetry = {0:None, 1:Symmetry2D(Nx,Ny)}[args.sym]



# create output folder and name the files
try:
    os.mkdir('pretraining_results/')
except FileExistsError:
    pass

bounds_x = "open"
bounds_y = "open"
filename = str(Nx)+"x"+str(Ny)+"_"+bounds_x+bounds_y+"_delta"+str(args.delta)+"_ps"+str(patchsize[0])+"x"+str(patchsize[1])+"_nsamples"
filename+= str(num_samples)+"_emb"+str(embedding_size)+"_nhead"+str(n_head)+"_nlayer"+str(n_layers)+"_dropout"+str(dropout)
if symmetry != None:
    filename += "_spatsym"
filename += "_J="+str(args.J)

#create model
device = initialize_torch()


# create model
model = TransformerModel(system_sizes, embedding_size, n_head, n_hid, n_layers, dropout=dropout, patchsize=patchsize, symmetry=symmetry, device=device, 
minibatch=minibatch)
num_params = sum([param.numel() for param in model.parameters()])
print('Number of parameters: ', num_params)




# load data for the pretraining
bases = ["Z", "X"]
if args.J < 0:
    mag   = "ferro"
else:
    mag = "antiferro"
exp_corrs = {}
exp_mag = {}
exp_samples = {}
for b in bases:
    """ here 
           - exp_samples = {"Z": (torch.tensor of size num_samples, Nx, Ny), "X": None},
           - exp_corrs = {"Z": None, "X": (torch.tensor of size Nx, Ny)}
           - and exp_mag = {"Z": None, "X": (torch.tensor of size Nx, Ny)}
        should be created.
    """

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --------- start the training ----------------------
E_training = []
Evar_training = []
C_training = []
epoch = 1
while epoch <= num_epochs:
    start = timeit.default_timer()
    cost  = cost_fct_corr(model, device, params, bounds_x, bounds_y, antisym, num_samples, exp_corrs=exp_corrs, true_mag=exp_mag)
    cost  += cost_fct_samples_ot(model, device, params, bounds_x, bounds_y, antisym, num_samples, exp_samples)
    optimizer.zero_grad()
    if epoch!=1:
        cost.backward() # Does backpropagation and calculates gradients
        optimizer.step()
    # save observables
    if epoch%5 == 0 or epoch == 1:
        with torch.no_grad():
            samples, weights = model.sample(num_samples)
            Elocs, log_probs, phases = get_Eloc(params,samples, model, bounds_x, bounds_y, antisym)
            Eloc = (weights * Elocs.detach()).sum()
            Eloc_var = torch.var(Elocs.detach())
        E_training.append(Eloc.cpu())
        Evar_training.append(Eloc_var.cpu())
        C_training.append(cost.detach().cpu())
        print('Epoch: {}/ {} ({}s) .............'.format(epoch, num_epochs, np.round(end-start,3)), end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var), end=', ')
        print("     mean(E/N): {:.8f}".format((Elocs/(Nx*Ny)).mean())+", var(E/N): {:.8f}".format((Elocs/(Nx*Ny)).var()))
        del Elocs, Eloc, Eloc_var, samples
    if epoch %10==0:
        torch.save(model.state_dict(), "pretrained_models/pretrained_model_"+filename+"_epoch="+str(epoch)+".pt")
        np.save("pretraining_results/Eloc_"+filename+".npy", np.array(E_training))
        np.save("pretraining_results/ElocVar_"+filename+".npy", np.array(Evar_training))
        np.save("pretraining_results/Cost_"+filename+".npy", np.array(C_training))
    end = timeit.default_timer()
    epoch += 1
    del cost
