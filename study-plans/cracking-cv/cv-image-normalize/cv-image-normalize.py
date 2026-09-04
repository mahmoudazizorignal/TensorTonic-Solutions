def normalize_image(image, mean, std):
    """
    Returns: 3D list of shape (H, W, C), each value rounded to 4 decimals
    """
    H = len(image)
    W = len(image[0])
    C = len(image[0][0])
    out = [[[0.0] * C for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            for c in range(C):
                v = (image[i][j][c] - mean[c]) / std[c]
                out[i][j][c] = round(v, 4)
    return out