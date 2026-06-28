import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pos = np.arange(0, seq_length).reshape(-1, 1)
    emb = np.arange(0, d_model).reshape(1, -1) // 2
    pos_enc = pos / (10000 ** (2 * emb / d_model))
    pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
    pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
    return pos_enc
