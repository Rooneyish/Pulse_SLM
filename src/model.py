import torch
import torch.nn as nn
from src.embeddings import Embedding, Positional_Embedding
from src.layers import Multi_Head_Attention, LayerNormalization, FeedForwardNetwork


class TransformerDecoderBlock(nn.Module):
    def __init__(self, num_heads, head_dim, hidden_n, dropout_prob=0.2):
        super().__init__()
        self.d_model = num_heads * head_dim

        self.mha = Multi_Head_Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_prob=dropout_prob,
        )

        self.ln_1 = LayerNormalization(d_model=self.d_model)
        self.ln_2 = LayerNormalization(d_model=self.d_model)

        self.ffn = FeedForwardNetwork(
            d_model=self.d_model, 
            hidden_n=hidden_n, 
            dropout_prob=dropout_prob
        )

    def forward(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        d_model, 
        num_layers, 
        num_heads, 
        head_dim, 
        seq_len, 
        hidden_n, 
        dropout_prob=0.2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.tok_emb = Embedding(vocab_size=vocab_size, d_model=d_model)
        self.pos_emb = Positional_Embedding(max_len=seq_len, d_model=d_model)
        self.emb_dropout = nn.Dropout(p=dropout_prob)

        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                num_heads=num_heads, 
                head_dim=head_dim, 
                hidden_n=hidden_n,
                dropout_prob=dropout_prob,
            ) for _ in range(num_layers)
        ])

        self.final_ln = LayerNormalization(d_model=d_model)

    def forward(self, x):
        tok = self.tok_emb(x)
        pos = self.pos_emb(x)
        
        x = self.emb_dropout(tok + pos)

        for block in self.decoder_blocks:
            x = block(x)

        x = self.final_ln(x)
        
        logits = x @ self.tok_emb.weight.T

        return logits