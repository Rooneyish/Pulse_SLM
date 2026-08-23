import math
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        return self.weight[x]

class Positional_Embedding(nn.Module):
    def __init__(self, max_len,d_model):
        super().__init__()
        self.max_len = max_len 
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)

        for pos in range(max_len):
            for i in range(d_model//2):
                denom = 10000**((2*i)/d_model)
                pe[pos, 2*i] = math.sin(pos/denom)
                pe[pos, 2*i+1] = math.cos(pos/denom)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.pe[:seq_len, :]    