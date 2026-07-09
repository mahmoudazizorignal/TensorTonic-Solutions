import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    N = token_ids.shape[0]
    words_pairs = torch.empty(0, 2, dtype=torch.int64)
    for i in range(N):
        target = token_ids[i]
        left = max(0, i - window)
        right = min(N - 1, i + window)
        cur_words_pairs = []
        for j in range(left, right + 1):
            if i == j: continue
            context = token_ids[j]
            cur_words_pairs.append([target, context])
        cur_words_pairs = torch.tensor(cur_words_pairs, dtype=torch.int64)
        words_pairs = torch.concat([words_pairs, cur_words_pairs], 0)
    return words_pairs
