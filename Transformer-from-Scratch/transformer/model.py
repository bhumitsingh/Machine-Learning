import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformer.layers.encoder import Encoder
from transformer.layers.decoder import Decoder

class Transformer(nn.Module):
    def __init__(self, 
                 src_vocab_size, 
                 tgt_vocab_size, 
                 d_model=512, 
                 num_heads=8, 
                 num_layers=6, 
                 d_ff=2048, 
                 max_len=5000, 
                 dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout
        )

        self.output_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: (batch_size, src_seq_len)
        tgt: (batch_size, tgt_seq_len)
        src_mask: (batch_size, 1, 1, src_seq_len)
        tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """

        # Encode source
        enc_output = self.encoder(src, src_mask)

        # Decode target with encoder context
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        # Project decoder output to vocab
        logits = self.output_linear(dec_output)
        return logits
