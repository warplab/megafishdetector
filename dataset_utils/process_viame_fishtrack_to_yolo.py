'''
WARNING: VIAME's FishTrack22 dataset is NOT standardized and in-flux, may need verification that nothing has changed when running this.
Very simple script to parse VIAME's FishTrack22 dataset into YOLO-style training directories

This script only supports single classes at the moment
TODO: support multi-class

Note that this COPIES all relevant data (including images!) into a new directory style

The data is only accessible through VIAME's interface at viame.kitware.org in the Collections/FishTrack or https://viame.kitware.com/#/collection/62afcb66dddafb68c8442126
'''

import sys
import os
import shutil
from os.path import join, isdir, exists
import pandas as pd
import glob

import cv2
import numpy as np
from tqdm import tqdm

# TODO: add flag for overwriting existing or not (to use the tmp.csv file)
INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

TMPCSV_FILENAME = "tmp.csv"

BLACKLIST_SPECIES = ["generic_object_proposal", "bait", "plant"]

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

#shutil.copytree(INPUT_DIR, OUTPUT_DIR, copy_function = shutil.copy2)

# Modify config YAML
with open(join(OUTPUT_DIR, "data.yaml"), "w") as f_out:
    f_out.write(f"path: ../data.yaml\n")
    f_out.write("train: images/training\n")
    f_out.write("val: images/validation\n")
    f_out.write("test: images/test\n")
    f_out.write("nc: 1\n")
    f_out.write("names: ['fish']")

# ===ITERATE THROUGH INPUT DATA===
# YOLO format is x,y,w,h -- where they are all normalized to range [0,1], and x, y are the CENTER of the bbox
#label_df.to_csv(label_filepath, sep=" ", header=None, index=None)

#TODO: For some reason this crashes the first time and I have no idea why...simply rerunning this again works.
if os.path.exists(join(OUTPUT_DIR, TMPCSV_FILENAME)):
    print("Loading existing datafiles...")
    dataset_df = pd.read_csv(join(OUTPUT_DIR, TMPCSV_FILENAME))
else:
    dataset_df = pd.DataFrame()
    print("Parsing dataset annotations files...")
    all_missing_img_names = []
    
    for dir_path in tqdm(glob.glob(join(INPUT_DIR,"*"))):
        #print(f"Processing {dir_path}")
        dir_name = os.path.basename(dir_path)
        #print(f"Base directory name: {dir_name}")
        
        # Some Viame CSVs are improperly formatted, we ignore that data here by just reading the first 11 columns
        subset_df = pd.read_csv(join(dir_path, "annotations.viame.csv"), usecols=list(range(11)))
        
        # The second row is just comments and not actual data
        subset_df = subset_df.iloc[1:,:]
        
        # TODO: this access method is crazy
        img_name_col = list(subset_df.columns).index("2: Video or Image Identifier")
        
        # Add image size information and copy all images over to the output directory
        subset_df['img_width'] = -1
        subset_df['img_height'] = -1

        missing_img_names = set()
        
        for img_name in pd.unique(subset_df.iloc[:,img_name_col]):
            src_img_path = join(dir_path, img_name)

            # Extraction sometimes results in mismatches, so we throw those out
            if not exists(src_img_path):
                missing_img_names.add(img_name)
                continue
            
            dst_img_name = f"{dir_name}.{img_name}"

            # TODO: why are there problematic images anywhere
            try:
                img = cv2.imread(src_img_path)
                img_width = img.shape[1]
                img_height = img.shape[0]
                subset_df.loc[subset_df.iloc[:,img_name_col] == img_name, 'img_width'] = img_width
                subset_df.loc[subset_df.iloc[:,img_name_col] == img_name, 'img_height'] = img_height
                
                # Copy image to the output directory
                shutil.copy2(src_img_path, join(OUTPUT_DIR, "images/training", dst_img_name))
            except:
                missing_img_names.add(img_name)
                print(f"Something wrong with image: , {dir_name}, {img_name}: {src_img_path}")
                continue

        missing_img_names_list = ",".join(list(missing_img_names))
        missing_img_names_str = f"{dir_name} -- {missing_img_names_list}"
        all_missing_img_names.append(missing_img_names_str)
        #print("Missing images: ", all_missing_img_names)
        for missing_img_name in missing_img_names:
            subset_df = subset_df.drop(subset_df[subset_df.iloc[:,img_name_col] == missing_img_name].index, axis=0)
            
        # Prepend the subdataset name to image names in case they are duplicated across datasets        
        subset_df.iloc[:, img_name_col] = dir_name + "." + subset_df.iloc[:, img_name_col].astype(str)
    
        # Add this to the overall dataset
        dataset_df = pd.concat((dataset_df, subset_df))

    print("All missing images: ", all_missing_img_names)
    
# Rename cols for easier processing
dataset_df.columns = ["det_id","img_name","frame_id","ux","uy","lx","ly","det_conf","target_len","species","conf","img_w","img_h"]
dataset_df.to_csv(join(OUTPUT_DIR, TMPCSV_FILENAME), index=False) # Make a tmp copy for testing

# ===FILTER DATA===
# Remove blacklisted species
species_col = list(dataset_df.columns).index('species')
dataset_df = dataset_df[~dataset_df.iloc[:,species_col].str.contains("|".join(BLACKLIST_SPECIES), case=False)]    

# ===CONVERT TO YOLO-STYLE FORMAT===
dataset_df["w"] = np.abs(dataset_df.lx - dataset_df.ux)
dataset_df["h"] = np.abs(dataset_df.uy - dataset_df.ly)

dataset_df["yolo_cx"] = ((dataset_df.ux+dataset_df.w/2)) / dataset_df.img_w
dataset_df["yolo_cy"] = ((dataset_df.uy+dataset_df.h/2)) / dataset_df.img_h
dataset_df["yolo_w"] = dataset_df.w / dataset_df.img_w
dataset_df["yolo_h"] = dataset_df.w / dataset_df.img_h

print(f"Species: {pd.unique(dataset_df['species'])}")
print(f"Number of bboxes found: {len(dataset_df.index)}")

num_images = len(pd.unique(dataset_df.img_name))
print(f"Number of images: {num_images}")
print(f"Saving bboxes...")
bad_images = []
for img_name in tqdm(pd.unique(dataset_df.img_name)):
    
    # Make groundtruth references
    src_img_path = join(OUTPUT_DIR, "images/training", img_name)

    if not exists(src_img_path):
        continue

    try:
        img = cv2.imread(src_img_path)
        img_bboxes = img.copy()
    except:
        bad_images.append(src_img_path)
        continue
    
    img_name_col = list(dataset_df.columns).index("img_name")

    for row_id, row in dataset_df[dataset_df.iloc[:,img_name_col] == img_name].iterrows():
        img_bboxes = cv2.rectangle(img_bboxes, (int(row[3]), int(row[4])), (int(row[5]), int(row[6])), (0,0,255))
    cv2.imwrite(join(OUTPUT_DIR, "images/groundtruth", img_name), img_bboxes)
    
    # Save the bboxes
    bboxes_df = dataset_df[dataset_df.img_name == img_name][["yolo_cx","yolo_cy","yolo_w","yolo_h"]]
    bboxes_df.insert(0, "class_label", 0)
    try:
        bboxes_df.to_csv(join(OUTPUT_DIR, "labels/training", img_name[:-4]+".txt"), sep=" ", header=None, index=None)
    except pd.errors.EmptyDataError:
        pass
    
print("Problematic images were found and skipped: ", bad_images)
