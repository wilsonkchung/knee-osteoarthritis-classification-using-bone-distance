# Steven

import cv2
import numpy as np
from skimage import morphology

def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Kien added: ( I had issues with cv2 reading images )
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    # -----------

    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Kien added:
    if bw is None:
        raise ValueError(f"Threshold failed on: {path}")
    # -----------

    return (bw > 0).astype(np.uint8)

def clean_mask(bw):
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=500)
    return bw.astype(np.uint8)