import torch

def create_padding_mask(seq, pad_token=0):
    """
    seq: (batch_size, seq_len)
    Returns: (batch_size, 1, 1, seq_len)
    """
    mask = (seq == pad_token).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
    return mask  # Broadcasted to (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    """
    Creates an upper-triangular matrix of -inf, with zeros on the diagonal.
    Returns: (1, 1, size, size)
    """
    mask = torch.triu(torch.ones((size, size)), diagonal=1)  # Upper triangular matrix
    return mask.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, size, size)

def combine_masks(padding_mask, look_ahead_mask):
    """
    Combines both masks for use in decoder's self-attention.
    If either is masked, the token is masked.
    """
    return (padding_mask.bool() | look_ahead_mask.bool())  # Ensure both masks are boolean

