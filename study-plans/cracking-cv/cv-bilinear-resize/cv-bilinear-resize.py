import numpy as np

def bilinear_resize(image, new_h, new_w):
    """
    Returns: list of lists of floats, shape (new_h, new_w), each value rounded to 4 decimals
    """
    image = np.asarray(image, dtype=float)
    h, w = image.shape[:2]
    
    h_ratio = (h - 1) / (new_h - 1) if new_h > 1 else 0.0
    w_ratio = (w - 1) / (new_w - 1) if new_w > 1 else 0.0
    
    y_src = np.arange(0, new_h, dtype=float).reshape(-1, 1) * h_ratio
    x_src = np.arange(0, new_w, dtype=float).reshape(1, -1) * w_ratio
    
    y_0 = np.floor(y_src).astype(int)
    x_0 = np.floor(x_src).astype(int)

    y_1 = np.minimum(y_0 + 1, h - 1).astype(int)
    x_1 = np.minimum(x_0 + 1, w - 1).astype(int)

    w_y = y_src - y_0
    w_x = x_src - x_0
    
    out = (1 - w_y) * (1 - w_x) * image[y_0, x_0] + (1 - w_y) * w_x * image[y_0, x_1] \
        + w_y * (1 - w_x) * image[y_1, x_0] + w_y * w_x * image[y_1, x_1]
    
    return np.round(out, decimals=4).tolist()
