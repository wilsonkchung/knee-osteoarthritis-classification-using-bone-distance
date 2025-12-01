# Steven

from load_masks import load_mask, clean_mask
from identify_bones import get_femur_tibia
from boundaries import extract_boundary
from distance import compute_distance

path = "./preds/9002116_050_pred.png"

bw = load_mask(path)
bw = clean_mask(bw)

femur, tibia = get_femur_tibia(bw)
if femur is None or tibia is None:
    print("invalid slice")
    exit()

f_b = extract_boundary(femur)
t_b = extract_boundary(tibia)

d = compute_distance(f_b, t_b)
print("distance:", d)
