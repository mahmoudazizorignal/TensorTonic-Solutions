import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    positions = np.array(np.arange(seq_length)).reshape(-1, 1)
    dimension = 10000 ** np.array(np.arange(d_model) / d_model).reshape(1, -1)

    embedding = positions / dimension
    embedding[:,  ::2] = np.sin(embedding[:,  ::2])
    embedding[:, 1::2] = np.cos(embedding[:, 1::2])

    return embedding