# megafishdetector

Detector for generic "fish" trained on publicly available datasets, currently supporting YOLO-style bounding boxes prediction and training.

Initial experiments to train a generic MegaFishDetector modelled off of the MegaDetector for land animals (https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)

Currently based on YOLOv5 (https://github.com/ultralytics/yolov5).

This repo contains links to public datasets, code to parse datasets into a common format (currently YOLO darknet only), and a model zoo for people to start with. For instructions to run, see the link above.

## Instructions
1. Install [Yolov5](https://github.com/ultralytics/yolov5)
2. Download desired network [weights](https://github.com/warplab/megafishdetector/blob/main/MODEL_ZOO.md)
3. Usage (from yolov5 root): python detect.py --imgsz 1280 --conf-thres 0.1  --weights [path/to/megafishdetector_v0_yolov5m_1280p] --source [path/to/video/image folder]

## Public Datasets Used in v0:

- [AIMs Ozfish](https://github.com/open-AIMS/ozfish) 
- [FathomNet](https://www.fathomnet.org/)
- [VIAME FishTrack](https://viame.kitware.com/#/collection/62afcb66dddafb68c8442126)
- [NOAA Puget Sound Nearshore Fish (2017-2018)](https://lila.science/datasets/noaa-puget-sound-nearshore-fish)
- [DeepFish](https://alzayats.github.io/DeepFish/)
- [NOAA Labelled Fishes in the Wild](https://www.st.nmfs.noaa.gov/aiasi/DataSets.html)

## To Cite:

[paper](https://arxiv.org/abs/2305.02330)
```
@misc{yang2023biological,
      title={Biological Hotspot Mapping in Coral Reefs with Robotic Visual Surveys}, 
      author={Daniel Yang and Levi Cai and Stewart Jamieson and Yogesh Girdhar},
      year={2023},
      eprint={2305.02330},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
## TODO:
- Train larger models
- requirements.txt for things like fathomnet environment
- COCO format output

