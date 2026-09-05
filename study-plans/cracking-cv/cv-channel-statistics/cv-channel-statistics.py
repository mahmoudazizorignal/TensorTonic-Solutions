def channel_statistics(batch):
    """
    Returns: dict with keys "mean" and "std", each a list of length C, with every entry rounded to 4 decimals.
    """
    import math
    
    B = len(batch)
    H = len(batch[0])
    W = len(batch[0][0])
    C = len(batch[0][0][0])
    mean = [0.0] * C
    std = [0.0] * C
    for c in range(C):
        
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    mean[c] += batch[b][h][w][c]
        
        mean[c] /= (B * H * W)
        for b in range(B):
            for h in range(H):
                for w in range(W):
                    std[c] += (batch[b][h][w][c] - mean[c]) ** 2
        std[c] = math.sqrt(std[c] / (B * H * W))
        
        mean[c] = round(mean[c], 4)
        std[c] = round(std[c], 4)
    
    return {
        "mean": mean,
        "std": std,
    }