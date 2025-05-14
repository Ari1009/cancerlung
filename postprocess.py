import cv2
import numpy as np
from skimage import measure
from config import MORPH_KERNEL_SIZE

def get_bboxes(pred_mask, threshold=0.5):
    # Threshold and clean mask
    pred_mask = (pred_mask > threshold).astype(np.uint8) * 255
    kernel = np.ones(MORPH_KERNEL_SIZE, np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    
    # Label regions
    labels = measure.label(pred_mask)
    regions = measure.regionprops(labels)
    
    # Extract bounding boxes
    bboxes = []
    for prop in regions:
        y1, x1, y2, x2 = prop.bbox
        bboxes.append(((x1, y1), (x2, y2)))  # (x1,y1), (x2,y2)
    
    return pred_mask, bboxes