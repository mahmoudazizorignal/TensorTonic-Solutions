def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    histogram = [0] * 256

    for i in range(len(image)):
        for j in range(len(image[i])):
            histogram[image[i][j]] += 1

    return histogram