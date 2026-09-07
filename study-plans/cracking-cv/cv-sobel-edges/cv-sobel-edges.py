import numpy as np

def sobel_edges(image):
    """
    Returns: dict with keys "gx", "gy", "magnitude", each a list of lists of shape (H, W), with every entry rounded to 4 decimals
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape[:2]
    G_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)
    padded_image = np.pad(image, pad_width=(1, 1))

    g_x = np.zeros((H, W), dtype=float)
    g_y = np.zeros((H, W), dtype=float)
    magnitude = np.zeros((H, W), dtype=float)
    for i in range(H):
        for j in range(W):
            region = padded_image[i:i+3, j:j+3]
            g_x[i, j] = np.dot(region.ravel(), G_x.ravel())
            g_y[i, j] = np.dot(region.ravel(), G_y.ravel())
            magnitude[i, j] = np.sqrt(g_x[i, j] ** 2 + g_y[i, j] ** 2)
    
    return {
        "gx": np.round(g_x, decimals=4),
        "gy": np.round(g_y, decimals=4),
        "magnitude": np.round(magnitude, decimals=4),
    }
