import os
import cv2
from os.path import join, isdir

def cocoxywh2yoloxywh(xywh, img_w, img_h):
    """
    COCO: top_left_x, top_left_y, width, height
    YOLO: center_x, center_y, width, height, relative to img height and width
    """
    x,y,w,h = xywh
    return [(x + (w/2))/img_w, (y + (h/2))/img_h, w/img_w, h/img_h]

def yoloxywh2cv2xyxy(xywh, img_w, img_h):
    """
    YOLO: center_x, center_y, width, height, relative to img height and width
    CV2: (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)
    """
    x,y,w,h = xywh
    # todo: is rounding correct?
    tl_x = round(x*img_w-(w*img_w/2))
    tl_y = round(y*img_h-(h*img_h/2))
    br_x = round(x*img_w+(w*img_w/2))
    br_y = round(y*img_h+(h*img_h/2))
    return ((tl_x, tl_y),(br_x, br_y))

def rectangle_yoloxywh(img, xywh, color=(0,0,255), thickness=4):
    img_w = img.shape[1]
    img_h = img.shape[0]
    tl, br = yoloxywh2cv2xyxy(xywh, img_w, img_h)
    img_r = cv2.rectangle(img, tl, br, color, thickness)
    return img_r
    
def check_and_makedirs(path):
    if not isdir(path):
        os.makedirs(path, exist_ok=True)
    
