from load_masks import load_mask, clean_mask
from split_objects import split_by_midline
from boundaries import get_boundary_coords
from distance import distance_nn

path = "../../preds/9005075_077_pred.png"

bw = load_mask(path)
bw = clean_mask(bw)

femur, tibia = split_by_midline(bw)

if femur.sum() == 0 or tibia.sum() == 0:
    print("invalid slice")
    raise SystemExit

f_pts = get_boundary_coords(femur)
t_pts = get_boundary_coords(tibia)

if f_pts is None or t_pts is None:
    print("invalid boundaries")
    raise SystemExit

d = distance_nn(f_pts, t_pts)
print("distance:", d)
