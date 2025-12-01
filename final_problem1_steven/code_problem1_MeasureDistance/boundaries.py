# Steven

import cv2
import numpy as np

def extract_boundary(mask):
    eroded = cv2.erode(mask, np.ones((3,3), np.uint8))
    return mask - eroded
