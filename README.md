# megafishdetector

Detector for generic "fish" trained on publicly available datasets, currently supporting YOLO-style bounding boxes prediction and training.

Initial experiments to train a generic MegaFishDetector modelled off of the MegaDetector for land animals (https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)

Currently based on YOLOv5 (https://github.com/ultralytics/yolov5).

This repo contains links to public datasets, code to parse datasets into a common format (currently YOLO darknet only), and a model zoo for people to start with. For instructions to run, see the link above.

Example usage (from yolov5 root): python detect.py --imgsz 1280 --conf-thres 0.1  --weights <path to megafishdetector_v0_yolov5m_1280p> --source <path to your video/image folder>

To Cite:

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
TODO:
- Train larger models
- requirements.txt for things like fathomnet environment
- COCO format output

