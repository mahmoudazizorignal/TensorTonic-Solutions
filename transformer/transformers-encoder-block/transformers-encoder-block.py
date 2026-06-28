import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    batch_size, seq_len, d_model = Q.shape
    q_W = Q @ W_q
    k_W = K @ W_k
    v_W = V @ W_v
    
    q_W = q_W.reshape(batch_size, seq_len, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    k_W = k_W.reshape(batch_size, seq_len, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    v_W = v_W.reshape(batch_size, seq_len, num_heads, d_model // num_heads).transpose(0, 2, 1, 3)
    
    heads = softmax(q_W @ k_W.transpose(0, 1, 3, 2) / np.sqrt(d_model // num_heads)) @ v_W
    heads = heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    return heads @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    return np.maximum(x @ W1 + b1, 0) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    x_dash = layer_norm(
        x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads),
        gamma1, beta1,
    )
    
    output = layer_norm(
        x_dash + feed_forward(x_dash, W1, b1, W2, b2),
        gamma2, beta2,
    )
    
    return output
