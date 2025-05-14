import cv2
import numpy as np
from config import IMAGE_SIZE, NORMALIZATION_RANGE

def preprocess_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Preserve aspect ratio
    height, width = image.shape
    scale = min(IMAGE_SIZE[0]/width, IMAGE_SIZE[1]/height)
    new_size = (int(width * scale), int(height * scale))
    resized = cv2.resize(image, new_size)
    
    # Pad to 512x512
    pad_x = IMAGE_SIZE[0] - resized.shape[1]
    pad_y = IMAGE_SIZE[1] - resized.shape[0]
    padded = cv2.copyMakeBorder(
        resized, 
        pad_y//2, pad_y - pad_y//2,
        pad_x//2, pad_x - pad_x//2,
        cv2.BORDER_CONSTANT, 
        value=0
    )
    
    # Normalize
    padded = (padded - 127.0) / 127.0  # [-1, 1]
    return np.expand_dims(padded, axis=-1), (height, width)