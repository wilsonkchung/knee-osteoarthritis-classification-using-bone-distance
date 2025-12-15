# Kien & Steven

import csv
import os

from load_masks2 import load_mask, clean_mask
from split_objects2 import split_by_midline
from boundaries2 import get_boundary_coords
from distance2 import distance_nn

script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.normpath(os.path.join(script_dir, "../..", "preds"))

out = []

for name in os.listdir(root):
    if not name.lower().endswith(".png"):
        continue
    p = os.path.join(root, name)

    #
    bw = load_mask(p)
    bw = clean_mask(bw)

    femur, tibia = split_by_midline(bw)

    if femur.sum() == 0 or tibia.sum() == 0:
        out.append((name, None))
        continue

    f_pts = get_boundary_coords(femur)
    t_pts = get_boundary_coords(tibia)

    d = distance_nn(f_pts, t_pts)

    out.append((name, d))


# write to CSV
with open("final_problem3_kien_steven/output_secondAttempt.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "distance"])

    for name, d in out:
        writer.writerow([name, d])
