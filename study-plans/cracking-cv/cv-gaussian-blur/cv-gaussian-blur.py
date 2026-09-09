import numpy as np

def gaussian_blur_2d(image, kernel_size, sigma):
    """
    Returns: 2D list of floats with shape (H, W), each entry rounded to 4 decimals
    """
    image = np.asarray(image, dtype=float)
    g_1d = np.exp(-(np.arange(0, kernel_size, dtype=float) - (kernel_size - 1)/2) ** 2 / (2 * sigma**2))
    g_2d = np.outer(g_1d, g_1d)
    g_2d_normalized = g_2d / np.sum(g_2d)
    
    H, W = image.shape[:2]
    padded_image = np.pad(image, pad_width=(kernel_size - 1) // 2)
    out = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            out[i, j] = np.dot(region.ravel(), g_2d_normalized.ravel())
    
    return np.round(out, decimals=4).tolist()
