import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="Lung Cancer Detection App", layout="wide")

# Load the model
@st.cache_resource
def load_prediction_model():
    return load_model('chest_cancer_model_fine_tuned.h5')

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((150, 150))  # Reduced size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# Function to make prediction
def predict(image):
    model = load_prediction_model()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    cancer_probability = np.max(prediction)
    is_cancerous = "Cancer Detected" if cancer_probability > 0.5 else "No Cancer Detected"
    return is_cancerous, cancer_probability

# Function to generate Grad-CAM heatmap
def compute_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(index=-3).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Adjusted axis
    conv_outputs = tf.squeeze(conv_outputs)  # Ensure correct dimensions
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)  # Avoid division by zero
    return heatmap.numpy()

# Function to overlay heatmap
def overlay_heatmap(image, heatmap):
    heatmap = np.array(heatmap)  # Ensure heatmap is a NumPy array
    heatmap = cv2.resize(heatmap, (image.width, image.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    image = np.array(image)
    superimposed_image = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(superimposed_image)

# Main function
def main():
    st.title("Lung Cancer Detection from CT Scan Images")
    uploaded_file = st.file_uploader("Upload a lung scan image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Make prediction
        result, confidence = predict(image)
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence Score: {confidence:.2%}")

        # Compute Grad-CAM heatmap
        image_tensor = preprocess_image(image)
        heatmap = compute_gradcam(load_prediction_model(), image_tensor)

        # Overlay heatmap on image
        highlighted_image = overlay_heatmap(image, heatmap)
        st.image(highlighted_image, caption="Highlighted Cancer Region", use_container_width=True)

if __name__ == "__main__":
    main()
