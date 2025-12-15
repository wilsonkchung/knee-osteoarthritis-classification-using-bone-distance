from skimage import measure
import numpy as np

def split_by_midline(bw):
    h, w = bw.shape
    mid = h // 2
    labels = measure.label(bw, connectivity=2)
    props = measure.regionprops(labels)

    femur = np.zeros_like(bw)
    tibia = np.zeros_like(bw)

    for p in props:
        cy = p.centroid[0]
        mask = (labels == p.label)
        if cy < mid:
            femur |= mask
        else:
            tibia |= mask

    femur = femur.astype(np.uint8)
    tibia = tibia.astype(np.uint8)
    return femur, tibia
