import glob
from load_masks import load_mask, clean_mask, enforce_slice_continuity, interpolate_missing_slices

def load_and_process(folder):
    paths = sorted(glob.glob(folder + "/*.png"))
    raw_masks = [load_mask(p) for p in paths]
    cleaned = [clean_mask(m) for m in raw_masks]
    connected = enforce_slice_continuity(cleaned)
    interpolated = interpolate_missing_slices(connected)
    return interpolated
