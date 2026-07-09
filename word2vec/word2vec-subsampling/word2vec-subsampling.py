import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    freq = counts / torch.sum(counts)
    return torch.minimum(torch.tensor(1), torch.sqrt(t / freq))
