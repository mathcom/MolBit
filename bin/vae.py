import math
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class CyclicalAnnealingScheduler(object):
    def __init__(self, T, M=4):
        super(CyclicalAnnealingScheduler, self).__init__()
        '''
        T : the total number of training iterations
        M : number of cycles
        R : proportin used to increase beta within a cycle
        R_inv : the inverse of R
        '''
        self.T = T
        self.M = M
        if M != 1:
            self.normalizer = T/M
            self.modulo = math.ceil(self.normalizer)
        else:
            self.normalizer = T
            self.modulo = T
        
    def __call__(self, step):
        tau = (step % self.modulo) / self.normalizer # 0 <= tau < 1
        beta = max(min(self._monotonically_increasing_ft(tau), 1.), 1e-4)
        return beta
        
    def _monotonically_increasing_ft(self, x):
        return 2. * x - 0.5
        

class SmilesVAE(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, latent_size, sos_idx, eos_idx, pad_idx, num_layers=2, device=None):
        super(SmilesVAE, self).__init__()
        
        ## params
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
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
        self.hidden2mean = nn.Linear(self.hidden_size * 2 * self.num_layers, self.latent_size)
        self.hidden2logvar = nn.Linear(self.hidden_size * 2 * self.num_layers, self.latent_size)
        self.latent2hidden = nn.Sequential(nn.Linear(self.latent_size, self.hidden_size * self.num_layers), nn.Tanh())
        self.output2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        
        ## Optimizer
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        ## device
        self.to(self.device)
        
        
    def partial_fit(self, smiles, length, beta=1.): # ex) smiles.shape = (batch, seq), length.shape = (batch, )
        ## Training phase
        self.train()
        
        ## Forward
        logp, mean, logvar, z = self(smiles, length)
        
        ## Loss
        loss_recon, loss_kl = self._loss_ft(smiles, logp, mean, logvar)
        loss_vae = loss_recon + beta * loss_kl
        
        ## Backpropagation
        self.optim.zero_grad()
        loss_vae.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.) # gradient clipping
        self.optim.step()
        
        return loss_vae.item(), loss_recon.item(), loss_kl.item()
        
        
    def _loss_ft(self, target, logp, mean, logvar):
        '''
        # target.shape = (batch, seq)
        # logp.shape = (batch, seq, vocab)
        # mean.shape = (batch, latent)
        # logvar.shape = (batch, latent)
        '''
        ## Reconstruction Loss
        target_ravel = target[:,1:].contiguous().view(-1) # target_ravel.shape = (batch*seq, )
        logp_ravel = logp[:,:-1,:].contiguous().view(-1, logp.size(2)) # logp_ravel.shape = (batch*seq, vocab)
        loss_recon = nn.NLLLoss(ignore_index=self.pad_idx, reduction="mean")(logp_ravel, target_ravel)

        ## KL Divergence Loss
        loss_kl = -0.5 * torch.sum(1. + logvar - mean.pow(2) - logvar.exp())

        return loss_recon, loss_kl
        
        
    def forward(self, inps, lens): # inps.shape = (batch, seq), lens.shape = (batch, )
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
        mean = self.hidden2mean(hiddens) # mean.shape = (batch, latent)
        logvar = self.hidden2logvar(hiddens) # logvar.shape = (batch, latent)
        std = torch.exp(0.5 * logvar) # std.shape = (batch, latent)
        epsilon = torch.randn([batch_size, self.latent_size], device=self.device) # epsilon.shape = (batch, latent)
        z = epsilon * std + mean # z.shape = (batch, latent)
        
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
        
        return logp, mean, logvar, z
        
        
    def inference(self, z=None, max_seqlen=100, greedy=False): # z.shape = (latent, )
        ## inference phase
        self.eval()
        
        ## Sampling
        if z is None:
            z = torch.randn([1, self.latent_size], device=self.device) # z.shape = (1, latent)
        elif torch.is_tensor(z) and z.dim() == 1:
            z = z.unsqueeze(0) # z.shape = (1, latent)
            
        ## Context
        contexts = self.latent2hidden(z) # contexts.shape = (1, hidden * numlayer)
        contexts = contexts.view(-1, self.hidden_size, self.num_layers) # contexts.shape = (1, hidden, numlayer)
        contexts = contexts.transpose(1,2).transpose(0,1).contiguous() # contexts.shape = (numlayer, 1, hidden)
        
        ## start
        inps_sos = torch.full(size=(1, 1), fill_value=self.sos_idx, dtype=torch.long, device=self.device) # inps_sos.shape = (1, 1)
        
        ## Generation
        generated = torch.zeros((1, max_seqlen), dtype=torch.long, device=self.device) # generated.shape = (1, max_seqlen)
        generated_logits = torch.zeros((1, max_seqlen, self.vocab_size), device=self.device) # generated_logits.shape = (1, max_seqlen, vocab)
        
        inps = inps_sos # inps.shape = (1, 1)
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
                inps = top_idx.contiguous().view(batch_size, 1) # inps.shape = (1, 1)
            else:
                probs = torch.softmax(logits, dim=-1) # probs.shape = (1, 1, vocab)
                inps = torch.multinomial(probs.view(probs.size(0), probs.size(2)), 1) # inps.shape = (1, 1)
            
        ## results
        results = generated[0].cpu().numpy() # results.shape = (max_seqlen, )
        return results
    
    
    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)


    def save_model(self, path):
        torch.save(self.state_dict(), path)
