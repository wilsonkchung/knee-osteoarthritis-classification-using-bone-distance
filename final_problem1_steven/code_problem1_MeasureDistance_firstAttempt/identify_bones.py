# Steven

from skimage import measure
import numpy as np

def get_femur_tibia(bw):
    labels = measure.label(bw, connectivity=2)
    props = measure.regionprops(labels)
    props = sorted(props, key=lambda x: x.area, reverse=True)
    if len(props) < 2:
        return None, None
    femur = (labels == props[0].label).astype(np.uint8)
    tibia = (labels == props[1].label).astype(np.uint8)
    return femur, tibia
