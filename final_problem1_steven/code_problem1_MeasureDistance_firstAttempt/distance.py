# Steven

import numpy as np

def compute_distance(femur_b, tibia_b):
    h, w = femur_b.shape
    distances = []
    for x in range(w):
        fy = np.where(femur_b[:, x] > 0)[0]
        ty = np.where(tibia_b[:, x] > 0)[0]
        if len(fy) == 0 or len(ty) == 0:
            continue
        d = ty.min() - fy.max()
        if d > 0:
            distances.append(d)
    if len(distances) == 0:
        return None
    return float(np.mean(distances))
