import sys
sys.path.insert(1, 'src')

# load general packages
import argparse
import os
import timeit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR

# load transformer module and helper functions
from model import TransformerModel
from helper import *
from symmetry import *



torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()
torch.set_default_tensor_type(torch.cuda.FloatTensor
                                if torch.cuda.is_available()
                                else torch.FloatTensor)

parser = argparse.ArgumentParser()
parser.add_argument("-Nx" , "--Nx"   ,  type=int,  default = 6 ,     help="length in x dir")
parser.add_argument("-Ny" , "--Ny"   ,  type=int,  default = 6 ,     help="length in y dir")
parser.add_argument("-J"  , "--J"    ,  type=float,default = 0.77 ,  help="J (spin interaction)")
parser.add_argument("-d"  , "--delta",  type=float,default = 1 ,     help="light shift delta")
parser.add_argument("-hd" , "--hd"   ,  type=int,  default = 32 ,    help="hidden dimension")
parser.add_argument("-nl" , "--nl"   ,  type=int,  default = 1 ,     help="number of TQS layers")
parser.add_argument("-nh" , "--nh"   ,  type=int,  default = 8 ,     help="number of TQS heads")
parser.add_argument("-sym", "--sym"  ,  type=int,  default = 0 ,     help="spatial symmetries (0:False, 1:True)")
parser.add_argument("-dmrg", "--dmrg" , type=int,  default = 0 ,   help="pretraining with DMRG (0:False, 1:True)")
parser.add_argument("-ts", "--training_samples", type=int,  default = 1000 , help="number of training samples")

args = parser.parse_args()
Nx     = args.Nx
Ny     = args.Ny
N_tot  = Nx * Ny
params = {"Jp": args.J, "delta": args.delta}
dmrg   = {0:False, 1:True}[args.dmrg]
training_samples = args.training_samples
system_sizes = [Nx,Ny]


# training hyperparameters
lr = 1e-4
num_epochs = 18000
minibatch = 1000
num_samples = 256*2
# for the Cosine learning rate schedule:
end = 20000 
maximum = 10000


# transformer parameters
patchsize=[2,2]
embedding_size = args.hd
n_head = args.nh
n_hid = embedding_size*5
n_layers = args.nl
dropout = 0
symmetry = {0:None, 1:Symmetry2D(Nx,Ny)}[args.sym]


# generate the results folder and name the files
try:
    os.mkdir('results/')
except FileExistsError:
    pass

bounds_x = "open"
bounds_y = "open"
filename = str(Nx)+"x"+str(Ny)+"_"+bounds_x+bounds_y+"_delta"+str(args.delta)+"_ps"+str(patchsize[0])+"x"+str(patchsize[1])+"_nsamples"
filename+= str(num_samples)+"_emb"+str(embedding_size)+"_nhead"+str(n_head)+"_nlayer"+str(n_layers)+"_dropout"+str(dropout)
if dmrg:
    filename += "_"+str(training_samples)+"DMRGsamples"
if symmetry != None:
    filename += "_spatsym"
filename += "_J="+str(args.J)

#create model and load the initialization obtained from the pretraining
device = initialize_torch()
E = np.load("pretraining_results/Eloc_"+filename+".npy")
start = [int(i*5/10)*10 for i,it in enumerate(np.real(E)) if it==min(np.real(E)[20:])][0]

print("pretrained_models/pretrained_model_"+filename+"_epoch="+str(start)+".pt")
if os.path.exists("pretrained_models/pretrained_model_"+filename+"_epoch="+str(start)+".pt"):
    print("load "+"pretrained_models/pretrained_model_"+filename+"_epoch="+str(start)+".pt")
    try:
        model = TransformerModel(system_sizes, embedding_size, n_head, n_hid, n_layers, dropout=dropout, patchsize=patchsize, symmetry=symmetry, U1sym=True, device=device, minibatch=minibatch)
        model.load_state_dict(torch.load("pretrained_models/pretrained_model_"+filename+"_epoch="+str(start)+".pt"), strict=True)
        print("Load initialization.")
    except RuntimeError:
        print("Model could not be loaded!!! Try with cpu")
        model = TransformerModel(system_sizes, embedding_size, n_head, n_hid, n_layers, dropout=dropout, patchsize=patchsize, symmetry=symmetry, U1sym=True, device="cpu", minibatch=minibatch)
        model.load_state_dict(torch.load("pretrained_models/pretrained_model_"+filename+"_epoch="+str(start)+".pt"))
        model.to(device)
num_params = sum([param.numel() for param in model.parameters()])
print('Number of parameters: ', num_params)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = CosineWarmupScheduler(optimizer, warmup=maximum, max_iters=end, start_iter=start) 



# --------- start the training ----------------------
E_training = []
Evar_training = []
epoch = 1
while epoch <= num_epochs-start:
    start_time = timeit.default_timer()    
    samples, weights = model.sample(num_samples)
    Elocs, cost, log_probs, phases  = cost_fct(samples, weights, model, device, params, bounds_x, bounds_y, antisym)
    optimizer.zero_grad()
    cost.backward() # Does backpropagation and calculates gradients
    optimizer.step()
    scheduler.step()
    
    # calculate observables to be saved and printed during the training
    Eloc = (weights * Elocs).sum().detach()
    Eloc_var = torch.var(Elocs).detach()
    E_training.append(Eloc.cpu())
    Evar_training.append(Eloc_var.cpu())
    if epoch%10 == 0 or epoch == 1:
        print('Epoch: {}/ {} ({}s) .............'.format(epoch, num_epochs, np.round(end_time-start_time,3)), end=' ')
        print("Loss: {:.8f}".format(cost)+", mean(E): {:.8f}".format(Eloc)+", var(E): {:.8f}".format(Eloc_var), end=', ')
        print("     mean(E/N): {:.8f}".format((Elocs/(Nx*Ny)).mean())+", var(E/N): {:.8f}".format((Elocs/(Nx*Ny)).var()))
    if (epoch+start)%1000==0 or epoch==100: #save intermediate versions
        print("save intermediate version.")
        torch.save(model.state_dict(), "models/model_"+filename+"_epoch="+str(epoch+start)+".pt")
        np.save("results/Eloc_"+filename+".npy", np.array(E_training))
        np.save("results/ElocVar_"+filename+".npy", np.array(Evar_training))
    end_time = timeit.default_timer()
    epoch += 1
    torch.cuda.empty_cache()
    del Elocs, Eloc, Eloc_var, samples, log_probs, phases, cost
np.save("results/Eloc_"+filename+".npy", np.array(E_training))
np.save("results/ElocVar_"+filename+".npy", np.array(Evar_training))
torch.save(model.state_dict(), "models/model_"+filename+".pt")



