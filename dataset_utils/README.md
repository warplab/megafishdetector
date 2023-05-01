## Lists/links of public fish datasets
- Lila: https://lila.science/otherdatasets#images-marine

- Awesome-fishies: https://github.com/dxyang/awesome-fishies

## Installation requirements
- python3
- fathomnet API
- pandas
- opencv-python

## Datasets available here and notes on utils
VIAME FISHTRACK22
1. Manually download viame fishtrack public dataset
2. Extract frames using extract_viame_fishtrack_videos.py to an intermediate directory
3. Run process_viame_fishtrack_to_yolo.py on the intermediate directory

FATHOMNET
1. Run download_and_process_fathomnet.py

AIMS OzFish
1. Download raw data
2. Run process_aims_ozfish_to_yolo.py

NOAA ESTUARY
1. Download raw data
2. Run process_noaa_estuary_to_yolo.py

NOAA Labelled Fishes In The Wild (LFITW)
1. Download raw data
2. Run process_lfitw_to_yolo.py (this cleans the data in a non-robust way, be warned!)

DeepFish
1. Download raw data
2. Run process_deepfish_to_yolo.py (this ONLY uses the segmentation data and transforms it into yolo bboxes)

Brackish Underwater
1. Unnecessary since yolov5 support single-class training now anyway, can download yoylo dataset directly

## Once downloaded, if training on YOLOv5
Recommend using the built-in yolovv5/utils/dataloaders.autosplit to split data into subsets (DeepFish already has predefined splits though)
