import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    d_model = K.shape[-1]
    return F.softmax(Q @ K.permute(0, 2, 1) / math.sqrt(d_model), dim=-1) @ V
