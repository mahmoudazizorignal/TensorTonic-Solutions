def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    """
    # Write code here
    if box_b[0] >= box_a[2] or box_b[1] >= box_a[3]:
        return 0
        
    top_left = [max(box_a[0], box_b[0]), max(box_a[1], box_b[1])]
    bottom_right = [min(box_a[2], box_b[2]), min(box_a[3], box_b[3])]

    intersection = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return intersection / (area_a + area_b - intersection)