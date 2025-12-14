from skimage.measure import find_contours
import numpy as np

def get_boundary_coords(mask):
    cs = find_contours(mask, 0.5)
    if not cs:
        return None
    pts = np.vstack(cs)
    return pts
