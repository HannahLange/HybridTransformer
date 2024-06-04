import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pos_encoding import TQSPositionalEncoding2D
from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
from torch.distributions.binomial import Binomial

pi = np.pi


"""
This code is adapted from Zhang et al. (https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075147) 
and Sprague et al. (https://www.nature.com/articles/s42005-024-01584-y)
"""

class Patch2D(nn.Module):
    def __init__(self,nx,ny,Lx,Ly,device):
        super().__init__()
        self.nx=nx
        self.ny=ny
        self.Ly=Ly
        self.Lx=Lx
        self.device = device

        #construct an index tensor for the reverse operation
        indices = torch.arange(Lx*Ly,device=device).unsqueeze(0)
        self.mixed_ = self.forward(indices).reshape([Lx*Ly])
        #inverse
        self.mixed=torch.argsort(self.mixed_)

    def forward(self,x,Lx=None,Ly=None):
        # type: (Tensor) -> Tensor
        nx,ny,Lx,Ly=self.nx,self.ny,self.Lx,self.Ly
        """Unflatten a tensor back to 2D, break it into nxn chunks, then flatten the sequence and the chunks
            Input:
                Tensor of shape [B,L]
            Output:
                Tensor of shape [B,L//n^2,n^2]
        """
        #make the input 2D then break it into 2x2 chunks 
        #afterwards reshape the 2x2 chunks to vectors of size 4 and flatten the 2d bit
        if Lx==None:
            nx,ny,Lx,Ly=self.nx,self.ny,self.Lx,self.Ly
        else:
            nx,ny=self.nx,self.ny
        return x.view([x.shape[0],Lx,Ly]).unfold(-2,nx,nx).unfold(-2,ny,ny).reshape([x.shape[0],int(Lx*Ly//(nx*ny)),nx*ny])

    def reverse(self,x,L=None):
        # type: (Tensor) -> Tensor
        """Inverse function of forward
            Input:
                Tensor of shape [B,L//n^2,n^2]
            Output:
                Tensor of shape [B,L]
        """
        Ly,Lx=self.Ly,self.Lx
        if L==None:
            L = Lx*Ly 
        mixed=torch.argsort(self.mixed_[:L])

        # Reversing is done with an index tensor because torch doesn't have an inverse method for unfold
        return x.reshape([x.shape[0],L])[:,mixed]


    def genpatch2onehot(self,patch,p):
        # type: (Tensor,int) -> Tensor
        """ Turn a sequence of size p patches into a onehot vector
        Inputs:
            patch - Tensor of shape [?,p]
            p (int) - the patch size
        """
        #moving the last dimension to the front
        patch = torch.round(patch.unsqueeze(0).transpose(-1,0).squeeze(-1)).to(torch.int64)
        out=torch.zeros(patch.shape[1:],device=patch.device)
        for i in range(p):
             out+=patch[i]<<i
        return nn.functional.one_hot(out.to(torch.int64), num_classes=1<<p)



class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, system_size, embedding_size, n_head, n_hid=2048, n_layers=2, phys_dim=2, dropout=0.0, patchsize=[2,2], symmetry=None, U1sym=True, device=None, minibatch=None):
        super(TransformerModel, self).__init__()
        self.device = device
        self.minibatch = minibatch

        self.system_size = system_size  # (n_size, n_dim)
        assert len(self.system_size) == 2
        self.n = self.system_size[0] * self.system_size[1]
        self.n_size = 1
        self.n_dim = 2
        max_system_size = self.system_size[0] * self.system_size[1]  # (n_dim, )
        self.phys_dim = phys_dim

        self.symmetry = self.symmetry_settings(symmetry)
        self.U1sym = U1sym

        self.phys_dim = phys_dim

        self.n_head = n_head
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.dropout = dropout

        self.src_mask = None
        self.patchsize = patchsize
        self.p = self.patchsize[0] * self.patchsize[1]
        self.patched_system_size = [int(self.system_size[0]/self.patchsize[0]), int(self.system_size[1]/self.patchsize[1])]
        self.npatch = math.ceil(self.n/(self.patchsize[0]*self.patchsize[1]))

        self.options=torch.zeros([1<<self.p,self.p],device=self.device)
        tmp=torch.arange(1<<self.p,device=self.device)
        for i in range(self.p):
            self.options[:,i]=(tmp>>i)%2
        self.patch = Patch2D(self.patchsize[0],self.patchsize[1],math.ceil(self.system_size[0]/self.patchsize[0])*self.patchsize[0],math.ceil(self.system_size[1]/self.patchsize[1])*self.patchsize[1], self.device)

        self.tokenize=nn.Sequential(nn.Linear(max(self.phys_dim, self.p),embedding_size),nn.Tanh()).to(self.device)

        pos_encoder = TQSPositionalEncoding2D
        self.pos_encoder = pos_encoder(embedding_size, self.n_dim, device, dropout=dropout)

        encoder_layers = TransformerEncoderLayer(d_model=embedding_size, dim_feedforward=n_hid, nhead=n_head, dropout=dropout, device=self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers).to(device)
        if torch.cuda.device_count() > 1:
            self.transformer_encoder_ = nn.DataParallel(self.transformer_encoder, dim=1)
        self.embedding_size = embedding_size
        self.amp_head = nn.Sequential(nn.Linear(embedding_size,embedding_size), nn.ReLU(), nn.Linear(embedding_size,1<<self.p))  
        self.phase_head = nn.Sequential(nn.Linear(embedding_size,embedding_size), nn.ReLU(), nn.Linear(embedding_size,1<<self.p)) 
        self.init_weights()
        self.to(device)
        self.get_sublattice_config()

    def get_sublattice_config(self):
        config = torch.zeros((self.system_size[0],self.system_size[1]), device=self.device)
        for i in range(self.system_size[0]):
            for j in range(self.system_size[1]):
                if not ((i%2==0 and j%2==0) or (i%2==1 and j%2==1)): #sublattice
                    config[i][j] = 1
        self.sublattice_config = config.to(torch.long)


    def symmetry_settings(self,sym):
        if sym != None:
            sym.add_symmetry('rotation_180')
        return sym

    def wrap_spins(self, spins):
        """
            spins: (n, batch)
        """
        n, batch, _  = spins.shape
        if self.p>1:
            if n<self.npatch:
                src = torch.zeros(n+1, batch, self.p, device=self.device)
                src[1:] = spins #F.one_hot(spins.to(torch.int64), num_classes=self.p)
            else:
                src = torch.zeros(n, batch, self.p, device=self.device)
                src[1:] = spins[:-1].to(torch.float32)
        else:
            spins = torch.reshape(spins, (spins.shape[0], spins.shape[1]))
            if n<self.n:
                src = torch.zeros(n+1, batch, self.phys_dim, device=self.device)
                src[1:] = F.one_hot(torch.round(spins).to(torch.int64), num_classes=self.phys_dim)
            else:
                src = torch.zeros(n, batch, self.phys_dim, device=self.device)
                src[1:] = F.one_hot(torch.round(spins[:-1]).to(torch.int64), num_classes=self.phys_dim).to(torch.float32)
        return src

    def _generate_square_subsequent_mask(self,sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.tokenize[0].weight, -initrange, initrange)
        nn.init.zeros_(self.tokenize[0].bias)
        for i in [0,2]:
            nn.init.uniform_(self.amp_head[i].weight, -initrange, initrange)
            nn.init.zeros_(self.amp_head[i].bias)
            nn.init.uniform_(self.phase_head[i].weight, -initrange, initrange)
            nn.init.zeros_(self.phase_head[i].bias)


    @staticmethod
    def softsign(x):
        """
            Similar to Hibat-Allah, Mohamed, et al. Physical Review Research 2.2 (2020): 023358.
        """
        return 2 * pi * (1 + x / (1 + x.abs()))

    def forward(self, spins, cache, compute_cache, compute_phase=True):
        # src: (seq, batch, input_dim)
        phase = None
        src = self.wrap_spins(spins)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            mask = self._generate_square_subsequent_mask(len(src))
            self.src_mask = mask

        if self.minibatch is None:
            src = self.tokenize(src) * math.sqrt(self.embedding_size)  # (seq, batch, embedding)
            src = self.pos_encoder(src, self.system_size)  # (seq, batch, embedding)  
            if compute_cache:
                output, cache = self.next_with_cache(src, self.src_mask, cache)
            else:
                output = self.transformer_encoder(src, self.src_mask)  # (seq, batch, embedding)
            psi_output = output[1:]  # only use the physical degrees of freedom
            amp = F.log_softmax(self.amp_head(psi_output), dim=-1)  # (seq, batch, phys_dim)
            if compute_phase:
                phase = self.softsign(self.phase_head(psi_output))  # (seq, batch, phys_dim)
        else:
            batch = src.shape[1]
            minibatch = self.minibatch
            repeat = int(np.ceil(batch / minibatch))
            amp = []
            phase = []
            new_cache = []
            for i in range(repeat):
                src_i = src[:, i * minibatch:(i + 1) * minibatch]
                src_i = self.tokenize(src_i) * math.sqrt(self.embedding_size)  # (seq, batch, embedding)
                src_i = self.pos_encoder(src_i, self.patched_system_size)  # (seq, batch, embedding)  
                if compute_cache:
                    if cache != None: c = cache[i]
                    else: c = None
                    output_i, c = self.next_with_cache(src_i, self.src_mask, c)
                    if c != None: new_cache.append(c)
                    psi_output = output_i
                else:
                    if torch.cuda.device_count() > 1:
                        output_i = self.transformer_encoder_(src_i, mask=torch.cat([self.src_mask for d in range(torch.cuda.device_count())], axis=1))
                    else:
                        output_i = self.transformer_encoder(src_i, mask=self.src_mask)
                    psi_output = output_i  # only use the physical degrees of freedom
                amp_i = F.log_softmax(self.amp_head(psi_output), dim=-1)  # (seq, batch, phys_dim)
                amp.append(amp_i)
                if compute_phase:
                    phase_i = self.softsign(self.phase_head(psi_output))  # (seq, batch, phys_dim)
                    phase.append(phase_i)
                    del phase_i
                del output_i, amp_i, src_i, psi_output
            amp = torch.cat(amp, dim=1)
            if compute_phase:
                phase = torch.cat(phase, dim=1)
        return amp, phase, new_cache

    def next_with_cache(self,tgt,mask,cache=None,idx=-1):
        # type: (Tensor,Optional[Tensor],int) -> Tuple[Tensor,Tensor]
        """Efficiently calculates the next output of a transformer given the input sequence and 
        cached intermediate layer encodings of the input sequence

        Inputs:
            tgt - Tensor of shape [L,B,Nh]
            cache - Tensor of shape ?
            idx - index from which to start
        Outputs:
            output - Tensor of shape [?,B,Nh]
            new_cache - Tensor of shape ?
        """
        output = tgt
        new_token_cache = []
        #go through each layer and apply self attention only to the last input
        for i,layer in enumerate(self.transformer_encoder.layers):
            tgt=output
            #have to merge the functions into one
            src = tgt[idx:, :, :]
            # self attention part
            src2 = layer.self_attn(
                src, #only do attention with the last elem of the sequence
                tgt,
                tgt,
                attn_mask=mask[idx:],  
                key_padding_mask=None,
            )[0]
            #straight from torch transformer encoder code
            src = src + layer.dropout1(src2)
            src = layer.norm1(src)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
            src = src + layer.dropout2(src2)
            src = layer.norm2(src)

            output = src
            new_token_cache.append(output)
            if cache is not None:
                #layers after layer 1 need to use a cache of the previous layer's output on each input
                output = torch.cat([cache[i], output], dim=0)

        #update cache with new output
        if cache is not None:
            new_cache = torch.cat([cache, torch.stack(new_token_cache, dim=0)], dim=1)
        else:
            new_cache = torch.stack(new_token_cache, dim=0)
        return output, new_cache

    @torch.jit.export
    @torch.no_grad()
    def sample(self, batch=10000, unique=True):
        """

        Parameters
        ----------
        batch : int, optional
            Number of samples to generate. The default is 10000.

        Returns
        -------
        samples : (n, batch)
            sampled binary configurations

        """
        batch0 = batch
        assert self.phys_dim == 2, "Only spin 1/2 systems are supported"
        samples = torch.zeros(0, batch, self.p, device=self.device)
        cache = None
        compute_cache = True
        for i in range(self.npatch): 
            log_amp, _, cache = self.forward(samples, cache, compute_cache, compute_phase=False)  # (seq, batch, phys_dim)
            amp = log_amp[-1].to(torch.float64).exp()  # (batch, phys_dim)
            if self.U1sym and i>0:
                samples_ = torch.permute(samples.clone(), (1,0,2))
                samples_ = self.patch.reverse(samples_,i*self.p)
                samples_ = torch.permute(samples_.clone(), (1,0))
                n_down = samples_.sum(dim=0)  # (batch, )
                n_up = samples_.shape[0] - n_down
                for d in range(amp.shape[-1]):
                    n_d = n_down + self.options[d].sum()
                    n_u = n_up + (self.p - self.options[d].sum())
                    mask = (n_d > self.n/2)
                    amp[mask, d] = 0 
                    mask = (n_u > self.n/2)
                    amp[mask, d] = 0
                    amp = torch.nn.functional.normalize(amp, p=1, eps = 1e-30)
            sampled_spins = self.options[torch.multinomial(amp, 1).squeeze(1)].reshape(1,batch,self.p)  # (batch, 1)
            samples = torch.cat([samples, sampled_spins], dim=0)
        if unique:
            samples, sample_count = torch.unique(samples, dim=1, return_counts=True)
        else:
            sample_count = torch.ones(batch0, device = self.device)
        samples = torch.permute(samples, (1,0,2))
        samples = self.patch.reverse(samples)
        if self.symmetry is not None:
            samples_sym = torch.permute(samples.clone(),(1,0))
            samples_sym = self.symmetry.apply_random(samples_sym)
            samples = torch.permute(samples_sym, (1,0))
        samples = torch.reshape(samples, (samples.shape[0],-1,self.system_size[1]))
        return samples, sample_count / batch0  # (n, batch), (batch, )

    @torch.jit.export
    def compute_psi(self, samples, check_duplicate=True, with_U1=False):
        """

        Parameters
        ----------
        samples : Tensor, (n, batch)
            samples drawn from the wave function
        check_duplicate : bool, optional
            whether to check for duplicate samples. The default is False.

        Returns
        -------
        log_amp : (batch, )
        log_phase : (batch, )

        extract the relevant part of the distribution, ignore the last output
        and the param distribution
        """
        samples_ = samples.clone()
        samples_ = torch.reshape(samples_, (samples.shape[0],-1))
        if self.symmetry is not None:
            samples_sym = torch.permute(samples_.clone(),(1,0))
            samples_, phase = self.symmetry(samples_sym)
            n_symm, n, batch0 = samples_.shape
            samples_sym = samples_
            samples_ = samples_.transpose(0, 1).reshape(n, -1)  # (n, n_symm*batch0)
            samples_ = torch.permute(samples_, (1,0))
        samples_sym = samples_
        if check_duplicate:
            samples_, inv_idx = torch.unique(samples_, dim=0, return_inverse=True)
        samples = self.patch.forward(samples_)
        samples = torch.permute(samples, (1,0,2))  # to (L, batch)
        n, batch, _ = samples.shape
        samples_ = self.patch.genpatch2onehot(samples,self.p)
        if with_U1 and self.U1sym:
            input = torch.zeros((0, batch, self.p), device=self.device)
            log_amp = []
            log_phase = []
            cache = None
            compute_cache = True
            for i in range(self.npatch):
                if i>0:
                    input = samples[:i]
                amp, phase, cache = self.forward(input, cache, compute_cache, compute_phase=True)  # (seq, batch, phys_dim)
                if self.U1sym and i>0:
                    n_down = torch.permute(samples[:i].clone(),(1,0,2)).reshape(samples_.shape[0],-1).sum(axis=1)
                    n_up = self.p*i - n_down
                    amp = amp[-1].to(torch.float64).exp()
                    a_s = []
                    for d in range(amp.shape[-1]):
                        n_d = n_down + self.options[d].sum()
                        n_u = n_up + (self.p - self.options[d].sum())
                        mask = (n_d > self.n/2)  
                        a = torch.where(mask, 0, amp[:, d])
                        mask = (n_u > self.n/2)  
                        a = torch.where(mask, 0, a)
                        a_s.append(a)
                    amp = torch.stack(a_s, axis=-1)
                    amp = torch.nn.functional.normalize(amp, p=1, eps = 1e-30)
                else:
                    amp = amp[-1].to(torch.float64).exp()
                log_amp.append(amp)
                log_phase.append(phase[-1])
            log_amp = torch.stack(log_amp, axis=0)
            log_phase = torch.stack(log_phase, axis=0)
            log_amp = torch.log(torch.sum(samples_*log_amp,dim=-1)).sum(dim=0) 
        else:
            log_amp, log_phase, _ = self.forward(samples, cache=None, compute_cache=False, compute_phase=True)
            log_amp = log_amp.to(torch.float64)
            log_amp = (torch.sum(samples_*log_amp,dim=-1)).sum(dim=0) 
        log_phase = torch.sum(samples_*log_phase,dim=-1).sum(dim=0).to(torch.float64)
        if check_duplicate:
            log_amp = log_amp[inv_idx]
            log_phase = log_phase[inv_idx]
        if self.symmetry is not None:
            log_amp = log_amp.reshape(n_symm, batch0)
            log_phase = log_phase.reshape(n_symm, batch0)
            log_phase = (((torch.complex(log_amp,log_phase))/ 2).exp().mean(dim=0))
            print(log_phase.dtype)
            eps = 1e-30
            log_phase.imag[torch.abs(log_phase.imag)<eps] = torch.sign(log_phase.imag[torch.abs(log_phase.imag)<eps])*eps
            log_phase.real[torch.abs(log_phase.real)<eps] = torch.sign(log_phase.real[torch.abs(log_phase.real)<eps])*eps
            log_phase = log_phase.imag.atan2(log_phase.real) * 2  # (batch0, )
            log_amp = log_amp.exp().mean(axis=0).log()   # (batch0, )
        log_phase = (log_phase/2)%(2*pi)
        return log_amp, log_phase
