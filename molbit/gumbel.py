import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.nn.functional import gumbel_softmax # (logits, tau=1, hard=False, eps=1e-10, dim=-1)
from torch.distributions.gumbel import Gumbel # (loc, scale, validate_args=None)


'''
참고사이트

https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py

'''

class TauAnnealingScheduler(object):
    def __init__(self, tau_min=0.2, update_step=500, annealing_rate=1e-4):
        self.update_step = update_step # {500, 1000}
        self.annealing_rate = annealing_rate # {1e-5, 1e-4}
        self.tau_min = tau_min
        self.step = 0
        self.tau = 0
        
    def __call__(self):
        if self.step % self.update_step == 0:
            self.tau = max(self.tau_min, math.exp(-self.annealing_rate * self.step)) # tau : 0.2 ~ 1.0
        self.step += 1
        return self.tau
        
    @property
    def min(self):
        return self.tau_min


class GumbelSmilesVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, sos_idx, eos_idx, pad_idx, num_layers=2, device=None):
        super(GumbelSmilesVAE, self).__init__()
        
        ## params
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.categorical_size = 2 # 2: binary, >2: category
        self.num_layers = num_layers
        self.device = torch.device('cpu') if device is None else device
        
        ## special tokens
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        
        ## models
        self.encoder = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True)
        self.embedding_enc = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=self.pad_idx)
        self.embedding_dec = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=self.pad_idx)
        self.hidden2latent = nn.Linear(self.hidden_size * 2 * self.num_layers, self.latent_size * self.categorical_size)
        self.latent2hidden = nn.Sequential(nn.Linear(self.latent_size, self.hidden_size * self.num_layers), nn.Tanh())
        self.output2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        
        ## annealing scheduler
        self.tau = TauAnnealingScheduler()
        
        ## Optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        ## device
        self.to(self.device)

    
    def partial_fit(self, smiles, length, beta=1.): # ex) smiles.shape = (batch, seq), length.shape = (batch, )
        ## Training phase
        self.train()
        
        ## Forward
        logp, q, z = self(smiles, length)
        
        ## Loss
        loss_recon, loss_kl = self._loss_ft(smiles, logp, q)
        loss_vae = loss_recon + beta * loss_kl
        
        ## Backpropagation
        self.optim.zero_grad()
        loss_vae.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.) # gradient clipping
        self.optim.step()
        
        return loss_vae.item(), loss_recon.item(), loss_kl.item()
        
    
    def _loss_ft(self, target, logp, q):
        '''
        # target.shape = (batch, seq)
        # logp.shape = (batch, seq, vocab)
        # q.shape = (batch, latent, category)
        '''
        ## Reconstruction Loss
        target_ravel = target[:,1:].contiguous().view(-1) # target_ravel.shape = (batch*seq, )
        logp_ravel = logp[:,:-1,:].contiguous().view(-1, logp.size(2)) # logp_ravel.shape = (batch*seq, vocab)
        loss_recon = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(logp_ravel, target_ravel)

        ## KL Divergence Loss
        logq = torch.log(q * self.categorical_size + 1e-10) # logq.shape = (batch, latent, category)
        loss_kl = torch.sum(q * logq, dim=-1).mean()

        return loss_recon, loss_kl
        
        
    def forward(self, inps, lens, hard=False): # inps.shape = (batch, seq), lens.shape = (batch, )
        batch_size = inps.size(0)
        
        ## Sorting by seqlen
        sorted_seqlen, sorted_idx = torch.sort(lens, descending=True)
        sorted_inps = inps[sorted_idx]
        
        ## Packing for encoder
        inps_emb = self.embedding_enc(sorted_inps) # inps_emb.shape = (batch, seq, emb)
        packed_inps = rnn_utils.pack_padded_sequence(inps_emb, sorted_seqlen.data.tolist(), batch_first=True)
        
        ## Encoding
        _, hiddens = self.encoder(packed_inps) # hiddens.shape = (2 * numlayer, batch, hidden)
        
        ## Latent vector
        hiddens = hiddens.transpose(0,1).contiguous().view(batch_size,-1) # hiddens.shape = (batch, hidden * 2 * numlayer)
        y = self.hidden2latent(hiddens) # y.shape = (batch, latent * category)
        y = y.view(batch_size, self.latent_size, self.categorical_size) # y.shape = (batch, latent, category)
        if hard:
            z = gumbel_softmax(y, tau=self.tau.min, hard=True, eps=1e-10, dim=-1) # z.shape = (batch, latent, category)
        else:
            z = gumbel_softmax(y, tau=self.tau(), hard=False, eps=1e-10, dim=-1) # z.shape = (batch, latent, category)
        z = z[:,:,0].contiguous() # z.shape = (batch, latent)
        q = nn.functional.softmax(y, dim=-1) # q.shape = (batch, latent, category)
        
        ## Context vector
        contexts = self.latent2hidden(z) # contexts.shape = (batch, hidden * numlayer)
        contexts = contexts.view(-1, self.hidden_size, self.num_layers) # contexts.shape = (batch, hidden, numlayer)
        contexts = contexts.transpose(1,2).transpose(0,1).contiguous() # contexts.shape = (numlayer, batch, hidden)
        
        ## Packing for decoder - Teacher forcing
        inps_emb_dec = self.embedding_dec(sorted_inps) # inps_emb.shape = (batch, seq, emb)
        packed_inps_dec = rnn_utils.pack_padded_sequence(inps_emb_dec, sorted_seqlen.data.tolist(), batch_first=True)
        
        ## Decoding
        packed_outs, _ = self.decoder(packed_inps_dec, contexts)
        sorted_outs, _ = rnn_utils.pad_packed_sequence(packed_outs, batch_first=True) # outs.shape = (batch, seq, hidden)
        sorted_outs = sorted_outs.contiguous()
        
        ## Reordeing
        _, original_idx = torch.sort(sorted_idx, descending=False)
        outs = sorted_outs[original_idx] # outs.shape = (batch, seq, hidden)
        
        ## Prediction
        logits = self.output2vocab(outs) # logits.shape = (batch, seq, vocab)
        logp = nn.functional.log_softmax(logits, dim=-1) # logp.shape = (batch, seq, vocab)
        
        return logp, q, z
        
    
    def inference(self, z=None, max_seqlen=100, greedy=False): # z.shape = (latent, )
        ## inference phase
        self.eval()
        
        ## Sampling
        if z is None:
            z = self.sample_z() # z.shape = (1, latent)
        elif torch.is_tensor(z) and z.dim() == 1:
            z = z.unsqueeze(0) # z.shape = (1, latent)
            
        ## Context
        contexts = self.latent2hidden(z) # contexts.shape = (1, hidden * numlayer)
        contexts = contexts.view(-1, self.hidden_size, self.num_layers) # contexts.shape = (1, hidden, numlayer)
        contexts = contexts.transpose(1,2).transpose(0,1).contiguous() # contexts.shape = (numlayer, 1, hidden)
        
        ## start token
        inps_sos = torch.full(size=(1,1), fill_value=self.sos_idx, dtype=torch.long, device=self.device) # inps_sos.shape = (1, 1)
        
        ## Generation
        generated = torch.zeros((1, max_seqlen), dtype=torch.long, device=self.device) # generated.shape = (1, max_seqlen)
        generated_logits = torch.zeros((1, max_seqlen, self.vocab_size), device=self.device) # generated_logits.shape = (1, max_seqlen, vocab)
        
        inps = inps_sos # inps.shape = (1,1)
        hiddens = contexts # hiddens.shape = (numlayer, 1, hidden)
        seqlen = 0
        for i in range(max_seqlen):
            ## Embedding
            inps_emb_dec = self.embedding_dec(inps) # inps_emb_dec.shape = (1, 1, emb)
            
            ## Decoding
            outs, hiddens = self.decoder(inps_emb_dec, hiddens) # outs.shape = (1, 1, hidden), hiddens = (numlayer, 1, hidden)
            
            ## Logit
            logits = self.output2vocab(outs) # logits.shape = (1, 1, vocab)
            
            ## Save
            generated[:,i] = inps.view(-1)
            generated_logits[:,i,:] = logits[:,0,:]
            seqlen += 1
            
            ## Terminal condition
            if inps[0][0] == self.eos_idx:
                break
            
            ## Next word
            if greedy:
                _, top_idx = torch.topk(logits, 1, dim=-1) # top_idx.shape = (1, 1, 1)
                inps = top_idx.contiguous().view(1, 1) # inps.shape = (1, 1)
            else:
                probs = torch.softmax(logits, dim=-1) # probs.shape = (1, 1, vocab)
                inps = torch.multinomial(probs.view(1, -1), 1) # inps.shape = (1, 1)
            
        ## results
        results = generated[0].cpu().numpy() # results.shape = (max_seqlen, )
        return results
    
    
    def load_model(self, path):
        weights = torch.load(path, weights_only=True)
        self.load_state_dict(weights)


    def save_model(self, path):
        torch.save(self.state_dict(), path)


    def sample_z(self):
        z = np.random.binomial(1, 0.5, self.latent_size).astype(np.float32) # z.shape = (latent, )
        z = torch.from_numpy(z).to(self.device) # z.shape = (latent, )
        return z

    def save_configs(self, path):
        with open(path, 'w') as f:
            f.write(f'vocab_size,{self.vocab_size}\n')
            f.write(f'embedding_size,{self.embedding_size}\n')
            f.write(f'hidden_size,{self.hidden_size}\n')
            f.write(f'latent_size,{self.latent_size}\n')
            f.write(f'categorical_size,{self.categorical_size}\n')
            f.write(f'num_layers,{self.num_layers}\n')
            f.write(f'sos_idx,{self.sos_idx}\n')
            f.write(f'eos_idx,{self.eos_idx}\n')
            f.write(f'pad_idx,{self.pad_idx}\n')
