import torch
import torch.nn as nn
from transformer.layers.attention import MultiHeadAttention
from transformer.layers.feed_forward import FeedForward
from torch.nn import LayerNorm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout = 0.1):
        super().__init__()

        # Multi-Head Attention
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)

        # FeedForward Network
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        # Layer Normalization
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        # Multi-Head Attention with residual connection
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = self.norm1(x + attn_output) # Add residual connection and apply LayerNorm

        # Feedforward Network with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output) # Add residual connection and apply LayerNorm

        return x