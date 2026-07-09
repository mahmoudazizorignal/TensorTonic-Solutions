import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    return F.softplus(-center_vec @ pos_vec) + torch.sum(F.softplus(center_vec.view(1, -1) @ neg_vecs.transpose(1, 0)))
