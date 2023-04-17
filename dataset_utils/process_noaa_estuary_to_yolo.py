import sys
import os
from os.path import join, isdir, exists
from pathlib import Path
import json
import shutil
import utils
import numpy as np
import cv2
from tqdm import tqdm

"""
NOAA estuary COCO to YOLO-style format
"""

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# Setup output directories
if not isdir(OUTPUT_DIR):
    shutil.copytree(INPUT_DIR, OUTPUT_DIR, copy_function = shutil.copy2)
    
utils.check_and_makedirs(join(OUTPUT_DIR, "labels"))
utils.check_and_makedirs(join(OUTPUT_DIR, "groundtruth"))

# Parse COCO json
with open(join(INPUT_DIR, "noaa_estuary_fish.json"), "r") as f:
    j = json.load(f)

# Gather all the annotations
data = {}
print("Processing annotations")
for annot in tqdm(j["annotations"]):
    if annot["image_id"] in data:
        data[annot["image_id"]].append(annot)
    else:
        data[annot["image_id"]] = [annot]

# Associate images with annotations and save
print("Processing images")
for img_f in tqdm(j["images"]):
    img_id = img_f["id"]
    img_w = img_f["width"]
    img_h = img_f["height"]
    class_bboxes = np.array([])
    
    if img_id in data and "bbox" in data[img_id][0]:
        bboxes = np.array([utils.cocoxywh2yoloxywh(x["bbox"], img_w, img_h) for x in data[img_id]])
        classes = np.matrix([x["category_id"] for x in data[img_id]]).T
        class_bboxes = np.concatenate((classes, bboxes), axis=1)
        np.savetxt(join(OUTPUT_DIR, "labels", f"{img_f['file_name']}.txt"), class_bboxes, fmt="%i %f %f %f %f")
    else:
        np.savetxt(join(OUTPUT_DIR, "labels", f"{img_f['file_name']}.txt"), class_bboxes)
        
# Generate groundtruth images for manual verification
print("Generating groundtruth images")
for img_f in tqdm(j["images"]):
    img_id = img_f["id"]
    img_w = img_f["width"]
    img_h = img_f["height"]
    img = cv2.imread(join(INPUT_DIR, "JPEGImages", img_f['file_name']))
    
    if img_id in data and "bbox" in data[img_id][0]:
        bboxes = np.array([utils.cocoxywh2yoloxywh(x["bbox"], img_w, img_h) for x in data[img_id]])
        for bbox in bboxes:
            img = utils.rectangle_yoloxywh(img, bbox)

    cv2.imwrite(join(OUTPUT_DIR, "groundtruth", img_f['file_name']), img)
