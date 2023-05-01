import sys
import os
from os.path import join, isdir, exists
from pathlib import Path
import pandas as pd
import shutil
import utils
import numpy as np
import cv2
from tqdm import tqdm

"""
AIMS OzFish to YOLO, single fish class
"""

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Setup output directories
if not isdir(OUTPUT_DIR):
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, copy_function = shutil.copy2)
    
utils.check_and_makedirs(join(OUTPUT_DIR, "labels"))
utils.check_and_makedirs(join(OUTPUT_DIR, "groundtruth"))

# Parse metadata csv
metadata_df = pd.read_csv(join(OUTPUT_DIR, "frame_metadata.csv"))

# Gather all the annotations
data = {}
print("Processing annotations")
for row_id, row in metadata_df.iterrows():
    annot = list(row.loc[["x0","y0","x1","y1"]])
    if row["file_name"] in data:
        data[row["file_name"]].append(annot)
    else:
        data[row["file_name"]] = [annot]

# Associate images with annotations and save
num_skipped = 0
print("Processing images")
for img_name in tqdm(data):

    img = cv2.imread(join(OUTPUT_DIR, "frames", img_name))

    if img is None:
        print(f"Skipping {img_name} corrupted")
        num_skipped += 1
        continue
    
    img_w = img.shape[1]
    img_h = img.shape[0]
    class_bboxes = np.array([])

    bboxes = data[img_name]
    bboxes = utils.cv2xyxy2yoloxywh(bboxes, img_w, img_h)
    classes = np.zeros((bboxes.shape[0], 1)) #todo: we actually can do species-specific labels
    class_bboxes = np.concatenate((classes, bboxes), axis=1)
    np.savetxt(join(OUTPUT_DIR, "labels", f"{img_name}.txt"), class_bboxes, fmt="%i %f %f %f %f")
        
    # Generate groundtruth images for manual verification
    for bbox in bboxes:
       img = utils.rectangle_yoloxywh(img, bbox)
       
    # for bbox in data[img_name]:
    #     img = cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]), (0,0,255), 3)

    cv2.imwrite(join(OUTPUT_DIR, "groundtruth", img_name), img)

print(f"Skipped {num_skipped} frames...(WARNING: These may be included in the image set still)")
