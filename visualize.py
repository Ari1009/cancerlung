import cv2
import numpy as np

def overlay_mask(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    return cv2.addWeighted(image, 0.7, mask, 0.3, 0)

def draw_bboxes(image, bboxes):
    for bbox in bboxes:
        (x1, y1), (x2, y2) = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image