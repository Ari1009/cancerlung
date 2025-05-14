import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model globally once
MODEL_PATH = "UNet_model_v2.h5"
model = load_model(MODEL_PATH, compile=False)

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (512, 512))
    norm_image = (image - 127.0) / 127.0
    norm_image = np.expand_dims(norm_image, axis=(0, -1))  # Shape: (1, 512, 512, 1)
    return norm_image, image  # Return preprocessed & original grayscale

def predict_mask(preprocessed_image):
    pred = model.predict(preprocessed_image)[0, :, :, 0]
    binary_mask = (pred > 0.5).astype(np.uint8)
    return binary_mask * 255  # Convert to 0-255 mask

def get_bounding_boxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]
    return boxes

def draw_boxes(original_image, boxes):
    image_with_boxes = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image_with_boxes
