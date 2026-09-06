import numpy as np

def pad_and_center_crop(image, pad, crop_h, crop_w):
    """
    Returns: 2D list of lists of floats with shape (crop_h, crop_w), each rounded to 4 decimals
    """
    image = np.asarray(image, dtype=float)
    H, W = image.shape[:2]
    r_start = (H + 2 * pad - crop_h) // 2
    c_start = (W + 2 * pad - crop_w) // 2
    padded_image = np.pad(image, pad_width=(pad, pad), mode='constant', constant_values=0)
    
    return padded_image[r_start:r_start + crop_h, c_start:c_start + crop_w].tolist()
