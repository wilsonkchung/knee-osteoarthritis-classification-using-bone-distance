# Steven

import cv2
import numpy as np
from skimage import morphology, measure

def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # Kien added: ( I had issues with cv2 reading images )
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    # -----------

    # Preserve existing binary masks or soft-edged masks
    # Use Otsu to avoid destroying weak boundaries
    _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = (bw > 0).astype(np.uint8)

    return bw

def clean_mask(bw):
    # Softer small-object filter to avoid deleting thin bone slices
    bw = morphology.remove_small_objects(bw.astype(bool), min_size=100)

    # Close fragmentation in femur/tibia masks
    bw = morphology.binary_closing(bw, morphology.disk(3))

    # Restore small internal cavities
    bw = morphology.remove_small_holes(bw, area_threshold=2000)

    return bw.astype(np.uint8)

def enforce_slice_continuity(mask_slices):
    result = []
    prev_centroid = None

    for bw in mask_slices:
        labeled, num = measure.label(bw, return_num=True)
        if num == 0:
            result.append(np.zeros_like(bw, dtype=np.uint8))
            continue

        props = measure.regionprops(labeled)

        if prev_centroid is None:
            # Keep largest component on first valid slice
            props_sorted = sorted(props, key=lambda p: p.area, reverse=True)
            keep = props_sorted[0]
        else:
            # Select component closest to previous centroid
            keep = min(props, key=lambda p: np.linalg.norm(np.array(p.centroid) - np.array(prev_centroid)))

        mask = np.zeros_like(bw, dtype=np.uint8)
        mask[labeled == keep.label] = 1
        prev_centroid = keep.centroid

        result.append(mask)

    return result

def interpolate_missing_slices(masks):
    masks = [m.astype(np.uint8) for m in masks]
    n = len(masks)

    # Identify runs of empty slices
    empty = [np.sum(m) == 0 for m in masks]

    i = 0
    while i < n:
        if empty[i]:
            start = i - 1
            j = i
            while j < n and empty[j]:
                j += 1
            end = j

            if start >= 0 and end < n:
                # Linear interpolation between start and end masks
                start_mask = masks[start].astype(float)
                end_mask = masks[end].astype(float)
                length = end - start

                for k in range(1, length):
                    alpha = k / length
                    blended = ((1 - alpha) * start_mask + alpha * end_mask)
                    # Convert back to binary
                    masks[start + k] = (blended > 0.5).astype(np.uint8)

            i = end
        else:
            i += 1

    return masks
