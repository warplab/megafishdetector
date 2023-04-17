'''
Script to parse DeepFish Segmentation data into YOLOv5-style training data

Note that this COPIES all relevant data (including images!) into a new directory style

Raw data available here: https://alzayats.github.io/DeepFish/
'''

import sys
import os
import shutil
from os.path import join, isdir
import pandas as pd

import cv2
import numpy as np

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
INCLUDE_NEGATIVE_EXAMPLES = True

OUTPUT_ROOT_NAME = "DeepFish_YOLO"
TRAINING_DIR_NAME = "Segmentation"
TRAINING_FILENAME = "train.csv"
VALIDATION_FILENAME = "val.csv"

class dataset():
    def __init__(self):
        """
        training = {img_src: str, bbox: list of lists in cx, cy, w, h}
        """
        root_dir = ""
        training = {}
        val = {}
        test = {}
        
# Create YAML
# Output file structure is:
# | DeepFish_YOLO
# --| images
# ----| training
# ----| validation
# --| labels
# ----| training
# ----| validation

if not isdir(join(OUTPUT_DIR, "images/training")):
    os.makedirs(join(OUTPUT_DIR, "images/training"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "labels/training")):
    os.makedirs(join(OUTPUT_DIR, "labels/training"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "images/validation")):
    os.makedirs(join(OUTPUT_DIR, "images/validation"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "labels/validation")):
    os.makedirs(join(OUTPUT_DIR, "labels/validation"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "images/test")):
    os.makedirs(join(OUTPUT_DIR, "images/test"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "labels/test")):
    os.makedirs(join(OUTPUT_DIR, "labels/test"), exist_ok=True)

if not isdir(join(OUTPUT_DIR, "images/groundtruth")):
    os.makedirs(join(OUTPUT_DIR, "images/groundtruth"), exist_ok=True)
    
with open(join(OUTPUT_DIR, "dataset.yaml"), "w") as f_out:
    f_out.write(f"path: ../datasets/{OUTPUT_ROOT_NAME}\n")
    f_out.write("train: images/training\n")
    f_out.write("val: images/validation\n")
    f_out.write("test: images/test\n")
    f_out.write("nc: 1\n")
    f_out.write("names: ['fish']")

# ===PROCESS DATA===
# YOLO format is x,y,w,h -- where they are all normalized to range [0,1], and x, y are the CENTER of the bbox
# WARNING: This assumes that there are not overlapping or occluded (split up) objects, this is NOT a safe assumption for all images in this dataset, but it only seemed like an image or two have this issue
train_df = pd.read_csv(join(INPUT_DIR, TRAINING_DIR_NAME, "train.csv"))
val_df = pd.read_csv(join(INPUT_DIR, TRAINING_DIR_NAME, "val.csv"))
test_df = pd.read_csv(join(INPUT_DIR, TRAINING_DIR_NAME, "test.csv"))

dataset_names = ["training", "validation", "test"]
dfs = [train_df, val_df, test_df]

for dataset_ind, dataset_name in enumerate(dataset_names):
    df = dfs[dataset_ind]

    for ind, row in df.iterrows():
        img_path = join(INPUT_DIR, TRAINING_DIR_NAME, "images", row.ID + ".jpg")
        mask_path = join(INPUT_DIR, TRAINING_DIR_NAME, "masks", row.ID + ".png")
        img_name = row.frames

        print(f"Processing {img_path}")
        
        if row.labels == 0 and not INCLUDE_NEGATIVE_EXAMPLES:
            continue

        shutil.copy2(img_path, join(OUTPUT_DIR, f"images/{dataset_name}", img_name + ".jpg"))

        mask = cv2.imread(mask_path)
        cc = cv2.connectedComponents(mask[:,:,0], 8) #white, so doesn't matter which channel

        #num_objs = np.max(cc)
        num_objs = cc[0]
        
        img = cv2.imread(img_path)
        img_bboxes = img.copy()
        img_w = img.shape[1]
        img_h = img.shape[0]
        
        with open(join(OUTPUT_DIR, f"labels/{dataset_name}", img_name + ".txt"), "w") as f_out:
            for i in range(1, num_objs):
                contours, hierarchy = cv2.findContours(255*(cc[1] == i), 2, 1)
                cnt = np.concatenate(contours) # merge all contours
                x, y, w, h = cv2.boundingRect(cnt)
                
                yolo_x = ((int(x)+int(w)/2)) / int(img_w)
                yolo_y = ((int(y)+int(h)/2)) / int(img_h)
                yolo_w = int(w) / int(img_w)
                yolo_h = int(h) / int(img_h)
                
                f_out.write(f"0 {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")
                img_bboxes = cv2.rectangle(img_bboxes, (x, y), (x+w, y+h), (0, 0, 255))

            cv2.imwrite(join(OUTPUT_DIR, "images/groundtruth", img_name + ".jpg"), img_bboxes)
    
