import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """       
    g = np.array(g, dtype=float)
    if max_norm <= 0: 
        return g
    
    g_norm = np.linalg.norm(g)

    g[g_norm > max_norm] = g[g_norm > max_norm] * max_norm / g_norm

    return g
    