"""
Pooling and Normalization for Transformer Hidden States.

Implements attention-mask-aware mean pooling and L2 normalization
to convert sequence representations into fixed-dimensional document embeddings.
"""

from typing import Union
import numpy as np
import torch
import torch.nn.functional as F


def masked_mean_pooling(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Computes attention-mask-aware mean pooling over transformer hidden states.
    
    Formula:
        embedding = sum(hidden_state * mask, dim=1) / clamp(sum(mask, dim=1), min=1e-9)
        
    Args:
        last_hidden_state: Tensor of shape (batch_size, seq_len, hidden_dim).
        attention_mask: Tensor of shape (batch_size, seq_len) with 1 for tokens and 0 for padding.
        
    Returns:
        torch.Tensor: Pooled embeddings of shape (batch_size, hidden_dim).
    """
    # Expand attention mask to match hidden state dimension: (batch, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    
    # Sum token embeddings excluding padding
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
    
    # Sum mask counts per sequence, clamped to avoid division by zero
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    
    return sum_embeddings / sum_mask


def l2_normalize(
    embeddings: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-12,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Applies L2 normalization across the embedding dimension.
    
    Args:
        embeddings: Tensor or ndarray of shape (batch_size, hidden_dim) or (hidden_dim,).
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        L2-normalized embeddings of the same type and shape.
    """
    if isinstance(embeddings, torch.Tensor):
        return F.normalize(embeddings, p=2, dim=-1, eps=eps)
    elif isinstance(embeddings, np.ndarray):
        norms = np.linalg.norm(embeddings, ord=2, axis=-1, keepdims=True)
        norms = np.maximum(norms, eps)
        return embeddings / norms
    else:
        raise TypeError(f"Unsupported embedding type: {type(embeddings)}")

