from scipy.spatial import cKDTree
import numpy as np

def distance_nn(f_pts, t_pts):
    tree = cKDTree(t_pts)
    dists, _ = tree.query(f_pts)
    if len(dists) == 0:
        return None
    return float(np.mean(dists))
