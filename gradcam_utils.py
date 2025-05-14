import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model

def preprocess_image(uploaded_file):
    # Load and preprocess the image
    img = tf.image.decode_image(uploaded_file.read(), channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return tf.expand_dims(img, axis=0)

def compute_gradcam_with_box(model, img_array, layer_name=None):
    if layer_name is None:
        # Automatically pick last conv layer
        layer_name = [layer.name for layer in model.layers if 'conv' in layer.name][-1]

    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / (np.max(cam) + 1e-10)
    cam = cv2.resize(cam, (224, 224))

    # Stronger threshold to filter noise
    threshold = 0.6
    mask = np.uint8(cam > threshold)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    original_img = img_array[0].numpy()
    original_img = np.uint8(original_img * 255)

    if contours:
        # Sort contours by area and draw the biggest
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if cv2.contourArea(largest_contour) > 50:  # avoid tiny false positives
            cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return original_img

