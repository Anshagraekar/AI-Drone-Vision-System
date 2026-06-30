# geometry.py

import math

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ( (x1 + x2) / 2, (y1 + y2) / 2 )

def pixel_to_3d(x, y, depth, fx, fy, cx, cy):
    Z = depth
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return X, Y, Z

def calculate_distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )