import json
from pycocotools import mask as maskUtils

# image size
H, W = 100, 100

# --- Ground Truth JSON (2 objects as polygons) ---

ann_fg_1 = [[10, 10, 20, 10, 20, 20, 10, 20]]  # will have a match
ann_fg_2 = [[60, 60, 70, 60, 70, 70, 60, 70]]  # will have a match
ann_fg_3 = [[80, 80, 90, 80, 90, 90, 80, 90]]  # will NOT have a match
ann_bg_1 = [[40, 40, 50, 40, 50, 50, 40, 50]]  # will have a match
ann_bg_2 = [[25, 25, 35, 25, 35, 35, 25, 35]] # will NOT have a match

gt = {
    "images": [
        {
            "id": 1,
            "file_name": "image1.jpg",
            "width": W,
            "height": H,
        }
    ],
    "categories": [
        {"id": 10000, "name": "plant"},
        {"id": 10001, "name": "plant_background"},
    ],
    "annotations": [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 10000, # foreground
            "segmentation": ann_fg_1,
            "area": 100,
            "bbox": [10, 10, 10, 10],
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 10000, # foreground
            "segmentation": ann_fg_2,
            "area": 100,
            "bbox": [60, 60, 10, 10],
            "iscrowd": 0
        },
        {
            "id": 3,
            "image_id": 1,
            "category_id": 10001, # background
            "segmentation": ann_bg_1,
            "area": 100,
            "bbox": [40, 40, 10, 10],
            "iscrowd": 0
        },
        {
            "id": 4,
            "image_id": 1,
            "category_id": 10001, # background
            "segmentation": ann_bg_2,
            "area": 100,
            "bbox": [25, 25, 10, 10],
            "iscrowd": 0
        },
        {
            "id": 5,
            "image_id": 1,
            "category_id": 10000, # foreground
            "segmentation": ann_fg_3,
            "area": 100,
            "bbox": [80, 80, 10, 10],
            "iscrowd": 0
        }
    ]
}

import os
out_gt_path = os.path.join(os.path.dirname(__file__), "gt_nobackground.json")
with open(out_gt_path, "w") as f:
    json.dump(gt, f, indent=2)

print("Wrote gt_nobackground.json")


# --- Predictions JSON (2 TP foreground + 1 TP background + 1 FP, encoded as RLE) ---
poly_tp1_fg = ann_fg_1  # matches GT foreground id 1
poly_tp2_fg = ann_fg_2  # matches GT foreground id 2
# NOTE: ann_fg_3 at [80, 80] will NOT be detected (creates FN for foreground)
poly_fp = [[5, 5, 8, 5, 8, 8, 5, 8]]  # no GT nearby (true FP)
poly_tp1_bg = ann_bg_1  # matches GT background id 3

rle_tp1_fg = maskUtils.frPyObjects(poly_tp1_fg, H, W)[0]
rle_tp2_fg = maskUtils.frPyObjects(poly_tp2_fg, H, W)[0]
rle_fp = maskUtils.frPyObjects(poly_fp, H, W)[0]
rle_tp1_bg = maskUtils.frPyObjects(poly_tp1_bg, H, W)[0]

# pycocotools returns byte strings for counts, convert to str
rle_tp1_fg["counts"] = rle_tp1_fg["counts"].decode("utf-8")
rle_tp2_fg["counts"] = rle_tp2_fg["counts"].decode("utf-8")
rle_fp["counts"] = rle_fp["counts"].decode("utf-8")
rle_tp1_bg["counts"] = rle_tp1_bg["counts"].decode("utf-8")

preds = [
    {
        "image_id": 1,
        "category_id": 10000,
        "segmentation": rle_tp1_fg,
        "score": 0.95
    },
    {
        "image_id": 1,
        "category_id": 10000,
        "segmentation": rle_tp2_fg,
        "score": 0.95
    },
    {
        "image_id": 1,
        "category_id": 10000,
        "segmentation": rle_fp,
        "score": 0.40
    },
    {
        "image_id": 1,
        "category_id": 10000,
        "segmentation": rle_tp1_bg,
        "score": 0.85
    }
]

out_pred_path = os.path.join(os.path.dirname(__file__), "pred_nobackground.json")
with open(out_pred_path, "w") as f:
    json.dump(preds, f, indent=2)

print("Wrote pred_nobackground.json")
