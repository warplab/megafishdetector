# Alternative to the shell script
# Note: this script is extremely fragile (since the VIAME dataset is extremely non-standard)
# ONLY SUPPORTS INTEGER FPSES

import glob
from os.path import join, isdir, exists, basename
import os
import json
import sys
import shutil
from pathlib import Path
import cv2
import pandas as pd
import datetime
from tqdm import tqdm

INPUT_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

META_JSON = "meta.json"
ANNOTATIONS_CSV = "annotations.viame.csv"

# Used to convert frame time to frame number
DATE_FORMAT = "%H:%M:%S.%f"
FRAME_NAME_FORMAT = "frame%06d.png"
VIDEO_TIMESTAMP_COL_NAME = "2: Video or Image Identifier"
BBOX_COLS = list(range(3,7))

show_visualization = False

whitelist_seqs = None

blacklist_species = None

def timestamps_to_frame_names(timestamps, fps):
    # timestamps: array of timestamp strings
    # WARNING: only works with integer FPS and specific time formats at the moment
    assert(type(fps)==int)
    t0 = datetime.datetime.strptime("00:00:00.000000", DATE_FORMAT)
    frame_nums = [int((datetime.datetime.strptime(x, DATE_FORMAT)-t0).total_seconds()*fps) for x in timestamps]
    frame_names = [FRAME_NAME_FORMAT%(frame) for frame in frame_nums]
    return frame_names

for seq_path in tqdm(glob.glob(join(INPUT_DIR, "*"))):
    seq_name = Path(seq_path).stem
    seq_file = basename(seq_path)

    #print(f"Processing {seq_path}")

    if whitelist_seqs is not None and seq_name not in whitelist_seqs:
        continue

    is_video_dir = seq_path[-4:] == ".mp4"

    # Copy folders and shared files over
    outdir_path = join(OUTPUT_DIR, seq_name)
    if not exists(outdir_path):
        shutil.copytree(seq_path, outdir_path)
        
    # TODO: make this more robust
    if not is_video_dir:
        continue

    # If video, parse it and extract images and frame names
    
    # Extract FPS from meta.json
    with open(join(seq_path, META_JSON), "r") as f:
        props = json.load(f)
        annotation_fps = int(props['fps'])

    # Extract and save frame names from the annotations file
    viame_csv = pd.read_csv(join(seq_path, ANNOTATIONS_CSV), usecols=list(range(11)))
    img_name_col = list(viame_csv.columns).index(VIDEO_TIMESTAMP_COL_NAME)
    timestamps = viame_csv.iloc[1:, img_name_col] # ignore first line of comments
    frame_names = timestamps_to_frame_names(timestamps, annotation_fps)
    viame_csv.iloc[1:, img_name_col] = frame_names
    viame_csv.to_csv(join(outdir_path, ANNOTATIONS_CSV), index=False)

    # Extract and save frames
    video_path = join(seq_path, seq_file)
    cap = cv2.VideoCapture(video_path)
    in_frame_num = 0
    out_frame_num = 0
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    rel_fps = int(original_fps / annotation_fps)

    #print(f"source fps: {original_fps} vs. annotated fps: {annotation_fps} vs. relative_fps: {rel_fps}")
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # TODO: THIS FAILS TO GRAB THE LAST FRAME IN MANY CASES, NEED TO ACTUALLY LOOK UP TIMESTAMPS
        if (in_frame_num+1) % rel_fps == 0 or in_frame_num == 0:
            frame_name = FRAME_NAME_FORMAT%(out_frame_num)
            if show_visualization:
                gt_frame = frame.copy()
                bboxes = viame_csv[viame_csv[VIDEO_TIMESTAMP_COL_NAME] == frame_name].iloc[:,BBOX_COLS]
                
                for index, bbox in bboxes.iterrows():
                    tl_x, tl_y, br_x, br_y = bbox.astype(int)
                    gt_frame = cv2.rectangle(gt_frame, (tl_x,tl_y), (br_x,br_y), (0,255,0))
                    
                cv2.imshow("test", gt_frame)
                cv2.waitKey(1)
            
            cv2.imwrite(join(outdir_path, frame_name), frame)
            out_frame_num += 1
        
        in_frame_num += 1
    cap.release()

    #print("total frames", in_frame_num)
    #print("rel_fps", rel_fps)
    #print("orig_fps", original_fps)
    #print("annot_fps", annotation_fps)    

