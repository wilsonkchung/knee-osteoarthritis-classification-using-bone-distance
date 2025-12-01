# Kien & Steven

import csv
import os
from load_masks import load_mask, clean_mask
from identify_bones import get_femur_tibia
from boundaries import extract_boundary
from distance import compute_distance

script_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.normpath(os.path.join(script_dir, "../..", "preds"))

out = []

for name in os.listdir(root):
    if not name.lower().endswith(".png"):
        continue
    p = os.path.join(root, name)

    bw = load_mask(p)
    bw = clean_mask(bw)

    femur, tibia = get_femur_tibia(bw)
    if femur is None or tibia is None:
        out.append((name, None))
        continue

    f_b = extract_boundary(femur)
    t_b = extract_boundary(tibia)
    d = compute_distance(f_b, t_b)

    out.append((name, d))


# write to CSV
with open("final_problem3_kien_steven/output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "distance"])

    for name, d in out:
        writer.writerow([name, d])
