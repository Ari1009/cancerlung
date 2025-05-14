import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# --- Dice Coefficient and Loss ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# --- Load UNet model with custom objects ---
model = tf.keras.models.load_model(
    "UNet_model_v2.h5",
    custom_objects={
        'dice_coef_loss': dice_coef_loss,
        'dice_coef': dice_coef  # Add this line!
    }
)

# --- Load and preprocess image ---
image_path = "sample_ct_scan.png"  # Replace with your test image path
img = Image.open(image_path).convert("L")  # convert to grayscale
img = img.resize((512, 512))
img_array = np.array(img).astype(np.float32) / 255.0  # Normalize to [0,1]

# --- Prepare for prediction ---
input_image = np.expand_dims(img_array, axis=-1)     # Shape: (512, 512, 1)
input_image = np.expand_dims(input_image, axis=0)    # Shape: (1, 512, 512, 1)

# --- Predict mask ---
predicted_mask = model.predict(input_image)[0]       # Shape: (512, 512, 1)
predicted_mask = (predicted_mask > 0.5).astype(np.uint8)  # Thresholding

# --- Display result ---
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original CT Scan")
plt.imshow(img_array, cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Nodule Mask")
plt.imshow(predicted_mask.squeeze(), cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
