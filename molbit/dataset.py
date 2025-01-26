import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class SmilesEncoder(object):
    def __init__(self, filepath):
        super(SmilesEncoder, self).__init__()
        self.char2idx, self.idx2char = self.load_char2idx(filepath)
        self.sos_char = "<"
        self.eos_char = ">"
        self.pad_char = "?"
        self.vocab_size = len(self.idx2char)
        print(f"Vocabulary size (including 3 special tokens): {self.vocab_size}")
        
    @property
    def sos_idx(self):
        return self.char2idx[self.sos_char]
    
    @property
    def eos_idx(self):
        return self.char2idx[self.eos_char]

    @property
    def pad_idx(self):
        return self.char2idx[self.pad_char]        
        
    def encode(self, smiles, max_seqlen): # smiles : list of strings, max_seqlen : int
        res = np.zeros((len(smiles), max_seqlen), dtype=np.int64)
        for i, smi in enumerate(smiles):
            for j, c in enumerate(smi):
                res[i][j] = self.char2idx[c]
        return torch.Tensor(res).long()
        
    def decode(self, encoded, trim=True): # encoded : list of indices
        res = []
        for indices in encoded:
            smi = []
            for i in indices:
                if i == self.pad_idx:
                    break
                else:
                    smi.append(self.idx2char[i])
            smi = "".join(smi)
            if trim:
                smi = smi.replace(self.sos_char, "").replace(self.eos_char, "")
            res.append(smi)
        return res
        
    def load_char2idx(self, filepath):
        df = pd.read_csv(filepath, index_col=0)
        idx2char = df.iloc[:,0].values.tolist()
        char2idx = {c:i for i,c in enumerate(idx2char)}
        return char2idx, idx2char
        


class SmilesDataset(Dataset):
    def __init__(self, filepath, device=None):
        super(SmilesDataset, self).__init__()
        ## Loading
        self.df = pd.read_csv(filepath)
        print(f"Number of SMILES (raw): {len(self.df)}")
        ## Params
        self.num_smiles = len(self.df)
        self.max_seqlen = self.df["length"].max() + 2 # sos & eos
        print(f"Maximum of seqlen: {self.max_seqlen}")
        ## Vocabulary
        self.sos_char = "<"
        self.eos_char = ">"
        self.pad_char = "?"
        self.char2idx, self.idx2char = self._make_char2idx(self.df["smiles"])
        self.vocab_size = len(self.idx2char)
        print(f"Vocabulary size (including 3 special tokens): {self.vocab_size}")
        ## PyTorch
        self.device = torch.device('cpu') if device is None else device
        
    
    def __len__(self):
        return self.num_smiles
        
        
    def __getitem__(self, idx):
        batch_smiles = self.sos_char + self.df["smiles"][idx] + self.eos_char
        batch_length = len(batch_smiles)
        return {"smiles": batch_smiles,
                "length": batch_length}
        
    @property
    def sos_idx(self):
        return self.char2idx[self.sos_char]
    
    @property
    def eos_idx(self):
        return self.char2idx[self.eos_char]

    @property
    def pad_idx(self):
        return self.char2idx[self.pad_char]        
        
    def encode(self, smiles, max_seqlen): # smiles : list of strings, max_seqlen : int
        res = np.zeros((len(smiles), max_seqlen), dtype=np.int64)
        for i, smi in enumerate(smiles):
            for j, c in enumerate(smi):
                res[i][j] = self.char2idx[c]
        return torch.Tensor(res).long().to(self.device)
        
    def to_tensor(self, properties):
        return torch.Tensor(properties).to(self.device)
        
    def decode(self, encoded, trim=True): # encoded : list of indices
        res = []
        for indices in encoded:
            smi = []
            for i in indices:
                if i == self.pad_idx:
                    break
                else:
                    smi.append(self.idx2char[i])
            smi = "".join(smi)
            if trim:
                smi = smi.replace(self.sos_char, "").replace(self.eos_char, "")
            res.append(smi)
        return res
        
    def save_char2idx(self, filepath):
        with open(filepath, "w") as fout:
            fout.write(",char\n")
            for i, c in enumerate(self.idx2char):
                fout.write(f"{i},{c}\n")
        
    def _make_char2idx(self, smiles):
        char2idx = {self.pad_char:0, self.sos_char:1, self.eos_char:2}
        idx2char = [self.pad_char, self.sos_char, self.eos_char]
        t = 3
        for smi in smiles:
            for char in smi:
                if char not in char2idx:
                    char2idx[char] = t
                    idx2char.append(char)
                    t += 1
        return char2idx, idx2char
        