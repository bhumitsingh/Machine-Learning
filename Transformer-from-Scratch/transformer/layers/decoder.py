import torch
import torch.nn as nn
import math
from transformer.layers.decoder_layer import DecoderLayer
from transformer.layers.encoder import PositionalEncoding, TokenEmbedding

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)

        return self.norm(x)
