'''
Very simple script to parse Brackish fish/animals dataset into a single class

Note that this COPIES all relevant data (including images!) into a new directory style

Raw data available here: https://public.roboflow.com/object-detection/brackish-underwater/1
'''

import sys
import os
import shutil
from os.path import join, isdir
import pandas as pd
import glob

import cv2
import numpy as np

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

OUTPUT_ROOT_NAME = "Brackish_YOLO"

shutil.copytree(INPUT_DIR, OUTPUT_DIR, copy_function = shutil.copy2)

# Modify config YAML

with open(join(OUTPUT_DIR, "data.yaml"), "w") as f_out:
    f_out.write(f"path: ../datasets/{OUTPUT_ROOT_NAME}\n")
    f_out.write("train: images/training\n")
    f_out.write("val: images/validation\n")
    f_out.write("test: images/test\n")
    f_out.write("nc: 1\n")
    f_out.write("names: ['fish']")

# ===PROCESS DATA===
# YOLO format is x,y,w,h -- where they are all normalized to range [0,1], and x, y are the CENTER of the bbox

dataset_names = ["train", "valid", "test"]

for dataset_name in dataset_names:
    for label_filepath in glob.glob(join(OUTPUT_DIR, dataset_name, "labels", "*")):
        print(label_filepath)
        try:
            label_df = pd.read_csv(label_filepath, sep=" ", header=None)
            label_df.loc[:,0] = 0
            label_df.to_csv(label_filepath, sep=" ", header=None, index=None)
        except pd.errors.EmptyDataError:
            pass
