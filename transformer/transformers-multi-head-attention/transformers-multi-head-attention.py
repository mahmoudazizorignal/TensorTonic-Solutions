import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
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
