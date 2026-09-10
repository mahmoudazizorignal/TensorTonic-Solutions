import numpy as np

def adaptive_avg_pool_2d(image, output_h, output_w):
    """
    Returns: 2D Python list of shape (output_h, output_w), values rounded to 4 decimals
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape[:2]
    
    out = np.zeros((output_h, output_w), dtype=float)
    for i in range(output_h):
        start_h = (i * H) // output_h
        end_h = ((i + 1) * H) // output_h + (((i + 1) * H) % output_h != 0)
        for j in range(output_w):
            start_w = (j * W) // output_w
            end_w = ((j + 1) * W) // output_w + (((j + 1) * W) % output_w != 0)
            
            out[i, j] = np.mean(image[start_h:end_h, start_w:end_w])
    
    return np.round(out, decimals=4).tolist()
