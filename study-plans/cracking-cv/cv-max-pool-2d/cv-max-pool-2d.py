import numpy as np

def max_pool_2d(image, kernel_size, stride):
    """
    Returns: 2D list of shape (H_out, W_out), max-pooled values rounded to 4 decimals
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape[:2]
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    
    out = np.zeros((H_out, W_out), dtype=float)
    for i in range(H_out):
        a = i * stride
        for j in range(W_out):
            b = j * stride
            region = image[a:a+kernel_size, b:b+kernel_size]
            out[i, j] = np.max(region)
    
    return out.tolist()
