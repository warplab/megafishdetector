'''
Script to parse LABELED_FISH_IN_THE_WILD data into YOLOv5-darknet-style data,

Note that this COPIES all relevant data (including images!) into a new directory style

Raw data download here: https://swfscdata.nmfs.noaa.gov/labeled-fishes-in-the-wild/
'''

import sys
import os
import shutil
from os.path import join, isdir

import cv2

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

ROOT_NAME = "NOAA_LABELED_FISHES_IN_THE_WILD_YOLO"
TRAINING_DIR_NAME = "Training_and_validation/Positive_fish"
TRAINING_FILENAME = "Positive_fish_(ALL)-MARKS_DATA.dat"
TRAINING_DEDUPED_FILENAME = "Positive_fish_(ALL)-MARKS_DATA-DEDUPED.dat"

# Create YAML
# Output file structure is:
# | LABELED_FISH_IN_THE_WILD_YOLO
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

if not isdir(join(OUTPUT_DIR, "images/groundtruth")):
    os.makedirs(join(OUTPUT_DIR, "images/groundtruth"), exist_ok=True)
    
with open(join(OUTPUT_DIR, "dataset.yaml"), "w") as f_out:
    f_out.write(f"path: ../datasets/{ROOT_NAME}\n")
    f_out.write("train: images/training\n")
    f_out.write("val: images/training\n")
    f_out.write("nc: 1\n")
    f_out.write("names: ['fish']")

# ===PROCESS DATA===
# YOLO format is x,y,w,h -- where they are all normalized to range [0,1], and x, y are the CENTER of the bbox
# LFITW format is also x,y,w,h -- in pixels, where x, y are the top-left corner of the bbox

# ===DEDUP AND MERGE TRAINING DATA ROWS===
# There are duplicates, though documentation says there should not be? At least they're neighboring?
# Deal with duplicates in the data file
with open(join(INPUT_DIR, TRAINING_DIR_NAME, TRAINING_FILENAME), "r") as f_in:
    l = 0
    prev_img_name = ""
    accum_num_bboxes = 0
    accum_bboxes = []
    
    with open(join(INPUT_DIR, TRAINING_DIR_NAME, TRAINING_DEDUPED_FILENAME), "w") as f_out:
        for line in f_in:
        
            elements = [x.strip() for x in line.split(' ')]
            img_name = elements[0]
            num_bboxes = int(elements[1])

            if prev_img_name == "":
                accum_num_bboxes = num_bboxes
                accum_bboxes = elements[2:]
                
            elif prev_img_name == img_name:
                accum_num_bboxes += num_bboxes
                accum_bboxes.extend(elements[2:])
            
            else:
                out_row = [prev_img_name, str(accum_num_bboxes)]
                out_row.extend(accum_bboxes)
                out_row = " ".join(out_row)
                f_out.write(f"{out_row}\n")
                accum_num_bboxes = num_bboxes
                accum_bboxes = elements[2:]

            prev_img_name = img_name
            l += 1
            
        out_row = [prev_img_name, str(accum_num_bboxes)]
        out_row.extend(accum_bboxes)
        out_row = " ".join(out_row)
        f_out.write(f"{out_row}\n")



# Only the .dat file is used to come up with labels
# TODO: some of these are actually duplicate images...dataset documentation is incorrect, need to handle properly
with open(join(INPUT_DIR, TRAINING_DIR_NAME, TRAINING_DEDUPED_FILENAME), "r") as f_in:
    l = 0
    prev_img_name = ""
    
    for line in f_in:
        
        elements = line.split(' ')
        img_name = elements[0]
        num_bboxes = int(elements[1])

        with open(join(OUTPUT_DIR, "labels/training", img_name[:-4] + ".txt"), "w") as f_out:
            img_path = join(INPUT_DIR, TRAINING_DIR_NAME, img_name)
            print(f"Processing {img_path}")
            shutil.copy2(img_path, join(OUTPUT_DIR, "images/training", img_name))

            img = cv2.imread(img_path)
            img_bboxes = img.copy()
            img_w = img.shape[1]
            img_h = img.shape[0]

            for bbox in range(num_bboxes):
                x, y, w, h = [int(j) for j in elements[4*bbox+2:4*bbox+4+2]]
                
                yolo_x = ((int(x)+int(w)/2)) / int(img_w)
                yolo_y = ((int(y)+int(h)/2)) / int(img_h)
                yolo_w = int(w) / int(img_w)
                yolo_h = int(h) / int(img_h)
                f_out.write(f"0 {yolo_x} {yolo_y} {yolo_w} {yolo_h}\n")

                img_bboxes = cv2.rectangle(img_bboxes, (x, y), (x+w, y+h), (0, 0, 255))
            cv2.imwrite(join(OUTPUT_DIR, "images/groundtruth", img_name), img_bboxes)
            
        l += 1

