# megafishdetector

Initial experiments to train a generic MegaFishDetector modelled off of the MegaDetector for land animals (https://github.com/microsoft/CameraTraps/blob/main/megadetector.md)

Currently based on YOLOv5 (https://github.com/ultralytics/yolov5).

This repo contains links to public datasets, code to parse datasets into a common format (currently YOLO darknet only), and a model zoo for people to start with. For instructions to run, see the link above.

Example usage (from yolov5 root): python detect.py --imgsz 1280 --conf-thres 0.1  --weights <path to megafishdetector_v0_yolov5m_1280p> --source <path to your video/image folder>

To Cite:
...coming soon!

TODO:
- Train larger models
- requirements.txt for things like fathomnet environment
- COCO format output

