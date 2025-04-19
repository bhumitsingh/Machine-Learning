import torch
import torch.nn as nn
from transformer.layers.encoder_layer import EncoderLayer
from transformer.layers.embedding import TokenEmbedding, PositionalEncoding

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        super().__init__()

        # Token Embedding and Positional Encoding
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Stack of Encoder Layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Step 1: Get token embeddings
        x = self.embedding(x)
        
        # Step 2: Add positional encodings
        x = self.positional_encoding(x)
        
        # Step 3: Pass through all encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Final Layer Normalization
        x = self.norm(x)
        
        return x
