import math
import torch
import torch.nn as nn

class Multi_Head_Attention(nn.Module):
    def __init__(self, num_heads, head_dim, dropout_prob=0.1, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_model = num_heads * head_dim

        self.W_q = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.W_k = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.W_v = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.W_o = nn.Parameter(torch.empty(self.d_model, self.d_model))
        
        self.b_q = nn.Parameter(torch.empty(1,self.d_model))
        self.b_k = nn.Parameter(torch.empty(1,self.d_model))
        self.b_v = nn.Parameter(torch.empty(1,self.d_model))
        self.b_o = nn.Parameter(torch.zeros(1, self.d_model))
        
        nn.init.xavier_uniform_(self.W_k)
        nn.init.xavier_uniform_(self.W_q)
        nn.init.xavier_uniform_(self.W_v)
        nn.init.xavier_uniform_(self.W_o)
        
        nn.init.zeros_(self.b_k)
        nn.init.zeros_(self.b_q)
        nn.init.zeros_(self.b_v)
        nn.init.zeros_(self.b_o)

        self.attn_dropout = nn.Dropout(p=dropout_prob)
        self.resid_dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # calculating Q, K and V
        q = x @ self.W_q + self.b_q
        k = x @ self.W_k + self.b_k
        v = x @ self.W_v + self.b_v

        # applying head 
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        # calcuating scaled dot product attention
        qk = q @ k.transpose(-2,-1) # calucating QK
        # finally scaled dot prodcut 
        scaled_qk = qk / math.sqrt(self.head_dim) 
        # applying mask
        scaled_qk = scaled_qk.masked_fill(mask, float('-inf'))
        
        # softmax
        scores = torch.softmax(scaled_qk, dim= -1)
        scores = self.attn_dropout(scores)
        
        # context
        context = scores @ v
        context = context.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        output = context @ self.W_o + self.b_o 
        
        return self.resid_dropout(output)

class LayerNormalization(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        x_norm = (x-mean) / torch.sqrt(var + self.eps)
        output = self.gamma * x_norm + self.beta

        return output
        
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_n, dropout_prob=0.1):
        super().__init__()
        self.W_1 = nn.Parameter(torch.empty(d_model, hidden_n))
        self.b_1 = nn.Parameter((torch.empty(1, hidden_n)))
        self.W_2 = nn.Parameter(torch.empty(hidden_n, d_model))
        self.b_2 = nn.Parameter(torch.empty(1,d_model))

        nn.init.xavier_uniform_(self.W_1)
        nn.init.xavier_uniform_(self.W_2)
        nn.init.zeros_(self.b_1)
        nn.init.zeros_(self.b_2)
        
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        Z_1 = x @ self.W_1 + self.b_1
        hidden = torch.relu(Z_1)
        Z_2 = hidden @ self.W_2 + self.b_2
        return self.dropout(Z_2)