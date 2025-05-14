import streamlit as st
import cv2
import numpy as np
from PIL import Image
from utils import preprocess_image, predict_mask, get_bounding_boxes, draw_boxes

st.set_page_config(page_title="Lung Nodule Detection", layout="centered")

st.title("🫁 Lung Nodule Detection (U-Net)")
st.markdown("Upload a **CT scan image** to detect possible lung nodules.")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner("Analyzing with U-Net..."):
        preprocessed, gray = preprocess_image(image)
        mask = predict_mask(preprocessed)
        boxes = get_bounding_boxes(mask)
        output = draw_boxes(gray, boxes)

    st.image(output, caption="Nodule Detection", use_column_width=True)

    st.success(f"Detected {len(boxes)} possible nodules.")
