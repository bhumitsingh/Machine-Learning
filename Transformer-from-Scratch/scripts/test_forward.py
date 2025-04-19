import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformer.model import Transformer
from transformer.utils import create_padding_mask, create_look_ahead_mask, combine_masks


# ==== Dummy input ====
batch_size = 2
src_seq_len = 6
tgt_seq_len = 5
src_vocab_size = 100
tgt_vocab_size = 100
pad_token = 0

# Random source and target sequences (with some padding)
src = torch.tensor([
    [5, 8, 23, 45, 0, 0],
    [12, 3, 0, 0, 0, 0]
])  # shape: (2, 6)

tgt = torch.tensor([
    [1, 7, 4, 0, 0],
    [1, 9, 0, 0, 0]
])  # shape: (2, 5)

# ==== Instantiate Transformer ====
model = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=2,
    d_ff=2048
)

# ==== Create Masks ====
src_mask = create_padding_mask(src, pad_token=pad_token)      # (2, 1, 1, 6)
tgt_padding_mask = create_padding_mask(tgt, pad_token=pad_token)  # (2, 1, 1, 5)
look_ahead_mask = create_look_ahead_mask(tgt_seq_len)         # (1, 1, 5, 5)
tgt_mask = combine_masks(tgt_padding_mask, look_ahead_mask)   # (2, 1, 5, 5)

# ==== Forward Pass ====
logits = model(src, tgt, src_mask, tgt_mask)

print(f"Output shape: {logits.shape}")  # Expected: (2, 5, 100)
