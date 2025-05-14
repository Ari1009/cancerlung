import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Load test data from the specified local paths
testX = np.load("C:/luna-16/Luna16/preprocessed official/testX.npy").astype(np.float32)
testY = np.load("C:/luna-16/Luna16/preprocessed official/testY.npy").astype(np.float32)

# Normalize testX and binarize testY
testX = (testX - 127.0) / 127.0  # Scaling testX to [-1, 1]
testY = (testY > 127).astype(np.float32)  # Binarizing testY

# Print data types for verification
print("testX dtype:", testX.dtype)
print("testY dtype:", testY.dtype)






testX = np.reshape(testX, (len(testX), 512, 512, 1))
testY = np.reshape(testY, (len(testY), 512, 512, 1))

print(testX.shape)
print(testY.shape)




def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)








import tensorflow as tf

# Custom objects required for model loading
custom_objects = {'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}

# Load models from specified paths
model_c = tf.keras.models.load_model("C:/luna-16/Luna16/models/FPR_classifier_model.h5", custom_objects=custom_objects)
model = tf.keras.models.load_model("C:/luna-16/Luna16/models/UNet_model_v2.h5", custom_objects=custom_objects)

# Optionally, check model summaries to confirm successful loading
#model_c.summary()
model.summary()


# Directly run predictions without specifying a GPU device
pred = model.predict(testX)




import numpy as np
import cv2
import matplotlib.pyplot as plt

def display(true, pred, X, m=0, n=50):
    """
    Displays ground truth and predicted masks with original images.

    Parameters:
    true (np.array): Ground truth masks.
    pred (np.array): Predicted masks from the model.
    X (np.array): Original lung CT images.
    m (int): Start index for the display range.
    n (int): End index for the display range.
    """
    n = min(len(true), n)  # Ensure 'n' is within bounds of the true array
    num_images = n - m
    rows = num_images // 2 if num_images % 2 == 0 else (num_images // 2) + 1
    plt.figure(figsize=(20, 5 * rows))

    for i, (t, p, x) in enumerate(zip(true[m:n], pred[m:n], X[m:n])):
        t, p, x = np.squeeze(t), np.squeeze(p), np.squeeze(x)
        
        # Create overlay images
        groundtruth_overlay = cv2.addWeighted(x, 0.5, t, 0.5, 0)
        prediction_overlay = cv2.addWeighted(x, 0.5, p, 0.5, 0)

        # Plotting ground truth and prediction side by side
        plt.subplot(rows, 2, i + 1)
        plt.title("GroundTruth" + " " * 36 + "Prediction", fontsize=14)
        combined_img = np.hstack((groundtruth_overlay, prediction_overlay))
        plt.imshow(combined_img, cmap="bone")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


display(testY,pred,testX)






import copy
m = copy.deepcopy(np.squeeze(pred[1]))  # Use 'pred' instead of 'pred_c'
plt.imshow(m, cmap="gray")



import cv2
m2 = m.copy()
m2[10:12,10:20] = 1
plt.imshow(m2, cmap="gray")




from skimage import measure

# Binarize the mask `m`
m[m >= 0.5] = 255
m[m < 0.5] = 0
m = m.astype(np.uint8)

# Label connected regions in the mask
labels = measure.label(m)

# Extract properties of labeled regions
regions = measure.regionprops(labels)

# Example: Print area and bounding box for each region
for region in regions:
    print(f"Region area: {region.area}")
    print(f"Bounding box: {region.bbox}")


np.unique(labels)



bb, cc, dd = [], [], []

for prop in regions:
    B = prop.bbox  # Bounding box coordinates (min_row, min_col, max_row, max_col)
    C = prop.centroid  # Centroid coordinates (row, col)
    D = prop.equivalent_diameter  # Equivalent diameter for the region

    # Adjust bounding box with padding and ensure it stays within 512x512 bounds
    bb.append((
        (max(0, B[1] - 8), max(0, B[0] - 8)),  # Top-left corner with padding
        (min(B[3] + 8, 512), min(B[2] + 8, 512))  # Bottom-right corner with padding
    ))

    # Append centroid and equivalent diameter
    cc.append(C)
    dd.append(D)

# Outputs: bb (bounding boxes), cc (centroids), dd (diameters)
bb, cc, dd



m2 = m.copy()  # Copy the original mask to avoid altering it
for rect in bb:
    m2 = cv2.rectangle(m2, rect[0], rect[1], color=(255), thickness=2)  # Draw rectangle for each bounding box

plt.imshow(m2, cmap="gray")  # Display the modified mask with bounding boxes







import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Sample coordinates in cc
# Assuming cc should be a list of tuples in the form [(y1, x1), (y2, x2), ...]
# Let's mock a version of cc with a single point if data is limited
# Replace this with your actual data if cc has it
cc = [(10, 10)]  # Adjust this or replace with your real data

# Check if there are at least two points, otherwise use one or mock data
if len(cc) >= 2:
    x1, y1 = int(cc[0][1]), int(cc[0][0])
    x2, y2 = int(cc[1][1]), int(cc[1][0])
    d = math.sqrt((x2 - x1)*2 + (y2 - y1)*2)
    
    # Create a blank 512x512 image and mark points
    b = np.zeros((512, 512), dtype=np.uint8)
    b[y1, x1] = 255  # Mark first point
    b[y2, x2] = 255  # Mark second point
    
    plt.title(f"Distance between points: {d:.2f}")
else:
    x1, y1 = int(cc[0][1]), int(cc[0][0])
    d = 0  # No distance with only one point

    # Create a blank 512x512 image and mark a single point
    b = np.zeros((512, 512), dtype=np.uint8)
    b[y1, x1] = 255  # Mark only one point
    
    plt.title("Only one point available; no distance to compute")

# Display the marked image
plt.figure(figsize=(10, 10))
plt.imshow(b, cmap='gray')
plt.show()

# Print the distance if applicable
if len(cc) >= 2:
    print("Distance between points:", d)
else:
    print("Only one point available; no distance to compute.")





#final evaluatiion 


import os
import cv2
import copy
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import SimpleITK as stk
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.cluster import KMeans
from skimage import measure

# Define Paths for LUNA16 Data
# Correcting the PATH variable to include both levels of 'subset0'

# Path for the main subset0 directory in LUNA16 dataset
PATH = "C:/luna-16/Luna16/subset0/subset0/"  

# Example .mhd file in subset0
FILE = "C:/luna-16/Luna16/subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd" 

# Load the MHD file using SimpleITK
image = stk.ReadImage(FILE)
image_array = stk.GetArrayFromImage(image)

# Display basic information about the loaded data
print("Image Array Shape:", image_array.shape)
print("Image Spacing:", image.GetSpacing())
print("Image Origin:", image.GetOrigin())



# Path variable corrected for double 'subset0'
PATH = "C:/luna-16/Luna16/subset0/subset0/"
FILE = "1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"

def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    origin = np.array(list(mhdimage.GetOrigin()))
    space = np.array(list(mhdimage.GetSpacing()))
    return ct_scan, origin, space

# Custom Evaluation metrics for U-Net model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)





# Load the model using the new path and provided custom_objects dictionary
custom_objects = {'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss}
model = tf.keras.models.load_model("C:/luna-16/Luna16/models/UNet_model.h5", custom_objects=custom_objects)

# Read Custom CT-Scan (mhd file)
ct, origin, space = load_mhd(PATH + FILE)
print(ct.shape)  # Expected output: (161, 512, 512)

# Extract dimensions and normalize the CT scan
num_z, height, width = ct.shape
ct_norm = cv2.normalize(ct, None, 0, 255, cv2.NORM_MINMAX)  # Normalizing the CT scan

# Display a slice of the CT scan
plt.imshow(ct_norm[57], cmap="gray")



# Create CLAHE (Contrast Limited Adaptive Histogram Equalization) filter for contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE filter to each layer in the normalized CT scan
ct_norm_improved = [clahe.apply(layer.astype(np.uint8)) for layer in ct_norm]

# Display a slice of the CT scan after applying CLAHE
plt.imshow(ct_norm_improved[57], cmap="gray")





center_idx = len(ct_norm_improved) // 2  # Use the middle slice for central area
if center_idx < 1:
    raise ValueError("Insufficient CT scan slices for processing.")

# Select a valid slice range and ensure within bounds.
central_area = ct_norm_improved[center_idx][100:400, 100:400]

# Ensure reshaping correctly before fitting KMeans
if central_area.size == 0:
    raise ValueError("Central area extracted is empty. Check slice dimensions.")
    
kmeans = KMeans(n_clusters=2).fit(np.reshape(central_area, [np.prod(central_area.shape), 1]))
centroids = sorted(kmeans.cluster_centers_.flatten())
threshold = np.mean(centroids)
print("Calculated Threshold:", threshold)







lung_masks = []
kernel_small = np.ones((4, 4), np.uint8)
kernel_large = np.ones((13, 13), np.uint8)
kernel_medium = np.ones((8, 8), np.uint8)

for layer in ct_norm_improved:
    ret, lung_roi = cv2.threshold(layer, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations for noise reduction and region enhancement
    lung_roi = cv2.erode(lung_roi, kernel=kernel_small)
    lung_roi = cv2.dilate(lung_roi, kernel=kernel_large)
    lung_roi = cv2.erode(lung_roi, kernel=kernel_medium)

    labels = measure.label(lung_roi)
    regions = measure.regionprops(labels)
    good_labels = []
    
    for prop in regions:
        B = prop.bbox
        # Ensure bbox filtering matches dataset characteristics
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    
    lung_roi_mask = np.zeros_like(labels, dtype=np.uint8)
    
    for N in good_labels:
        lung_roi_mask[labels == N] = 1

    # Contour-based external filtering
    contours, hierarchy = cv2.findContours(lung_roi_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    external_contours = np.zeros_like(lung_roi_mask)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:  # External contours only
            area = cv2.contourArea(contours[i])
            if area > 518.0:
                cv2.drawContours(external_contours, contours, i, (1), -1)

    # Post-processing with dilate/erode for noise and hole filling
    external_contours = cv2.dilate(external_contours, kernel=kernel_small)
    external_contours = cv2.bitwise_not(external_contours)
    external_contours = cv2.erode(external_contours, kernel=np.ones((7, 7)))
    external_contours = cv2.bitwise_not(external_contours)
    external_contours = cv2.dilate(external_contours, kernel=np.ones((12, 12)))
    external_contours = cv2.erode(external_contours, kernel=np.ones((12, 12)))

    lung_masks.append(external_contours.astype(np.uint8))

# Visualize a sample mask from the generated lung masks
plt.imshow(lung_masks[len(lung_masks) // 2], cmap="gray")





extracted_lungs = []

for lung, mask in zip(ct_norm_improved, lung_masks):
    # Extract lung region by applying the mask
    extracted = cv2.bitwise_and(lung, lung, mask=mask)
    extracted_lungs.append(extracted)

# Visualize a sample extracted lung region
sample_index = len(extracted_lungs) // 2  # Use a dynamic index to avoid hardcoding
plt.figure(figsize=(8, 8))
plt.imshow(extracted_lungs[sample_index], cmap="bone")
plt.title(f"Extracted Lung Region - Slice {sample_index}")
plt.axis("off")
plt.show()


X = np.array(extracted_lungs)
X.shape

X = (X-127.0)/127.0
X = X.astype(np.float32)
X.dtype


print("Min:",X.min(),"\nMax:",X.max())

X = np.reshape(X, (len(X), 512, 512, 1))
X.shape


predictions = model.predict(X)
predictions.shape



# Display the data type, minimum, and maximum values of predictions
print("Dtype:", predictions.dtype, "\nMin:", predictions.min(), "\nMax:", predictions.max())

predictions[predictions>=0.5] = 255
predictions[predictions<0.5] = 0
predictions = predictions.astype(np.uint8)


pred = list(predictions)
pred = [np.squeeze(i) for i in pred]


plt.imshow(pred[61], cmap="gray")



bboxes = []
centroids = []
diams = []

for mask in pred:
    # Dilate the mask to enhance connected regions
    mask = cv2.dilate(mask, kernel=np.ones((5, 5)))
    
    # Label connected components
    labels = measure.label(mask, connectivity=2)
    regions = measure.regionprops(labels)
    
    bb = []
    cc = []
    dd = []
    
    for prop in regions:
        B = prop.bbox  # Bounding box (min_row, min_col, max_row, max_col)
        C = prop.centroid  # Centroid of the region
        D = prop.equivalent_diameter_area  # Diameter of a circle with same area
        
        # Adjust bounding box with padding
        bb.append(((max(0, B[1]-8), max(0, B[0]-8)),  # Top-left (x1, y1)
                   (min(B[3]+8, 512), min(B[2]+8, 512))))  # Bottom-right (x2, y2)
        cc.append(C)  # Centroid (y, x)
        dd.append(D)  # Diameter
    
    bboxes.append(bb)
    centroids.append(cc)
    diams.append(dd)

# Display bounding boxes from a specific range
bboxes[65:71]  # Adjust index range based on available data




centroids[65:71]


diams[65:71]


def display2(imgs, titles=None, cmap="bone"):
    n = len(list(imgs))
    r = n//3 if n%3==0 else (n//3)+1
    plt.figure(figsize=(25,int(8*r)))
    for i,img in enumerate(imgs):
        plt.subplot(r,3,i+1)
        if titles is not None:
            plt.title(titles[i])
        plt.imshow(img, cmap=cmap)

bs = []
mimgs = copy.deepcopy(extracted_lungs)

for i, (img, boxes) in enumerate(zip(mimgs, bboxes)):
    for rect in boxes:
        # Draw a rectangle on the image
        img = cv2.rectangle(img, rect[0], rect[1], (255), 2)

# Display modified images within the specified range
display2(mimgs[40:55], titles=[f"Image {i}" for i in range(40, 55)])






idx = 53 # Specify the image index
v = copy.deepcopy(extracted_lungs)  # Create a copy to preserve the original

plt.figure(figsize=(5, 5))

# Check if there are any bounding boxes for the given index
if bboxes[idx]:
    # Draw a rectangle for the first bounding box of the specified index
    i = cv2.rectangle(v[idx], bboxes[idx][0][0], bboxes[idx][0][1], (255), 2)

    # Draw a circle at the centroid for the first region of the specified index
    i = cv2.circle(i, (int(centroids[idx][0][1]), int(centroids[idx][0][0])), 2, (255), -1)

    # Display the modified image
    plt.imshow(i, cmap='gray')
    plt.title(f"Image Index {idx} with Bounding Box and Centroid")
else:
    # Display the original image if no bounding boxes are found
    plt.imshow(v[idx], cmap='gray')
    plt.title(f"Image Index {idx} (No Bounding Boxes Found)")

plt.axis('off')
plt.show()



diams[53][0]*space[0]






# Load the FPR classifier model from the correct path
fpr_model = tf.keras.models.load_model("C:/luna-16/Luna16/models/FPR_classifier_model.h5")

# Copy normalized images to preserve the original
originals = copy.deepcopy(ct_norm_improved)

# Prepare final bounding boxes list
final_boxes = []

# Iterate over each image and its bounding boxes
for i, (img, bbox) in enumerate(zip(originals, bboxes)):
    img_boxes = []  # List to store boxes for current image
    for box in bbox:
        x1 = box[0][0]
        y1 = box[0][1]
        x2 = box[1][0]
        y2 = box[1][1]
        
        # Adjust small bounding boxes to a minimum size of 50x50
        if abs(x1 - x2) <= 50 or abs(y1 - y2) <= 50:
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            x1 = max(x - 25, 0)
            x2 = min(x + 25, 512)
            y1 = max(y - 25, 0)
            y2 = min(y + 25, 512)
            imgbox = img[y1:y2, x1:x2]
        else:
            imgbox = img[y1:y2, x1:x2]
        
        # Append the cropped image box
        img_boxes.append(imgbox)
    
    # Append all boxes for current image to final_boxes
    final_boxes.append(img_boxes)

plt.figure(figsize=(3, 3))
num_images = len(final_boxes[53])  # Get the number of images in the current list
for i, img in enumerate(final_boxes[53]):
    plt.subplot(1, num_images, i + 1)  # Adjust the number of columns dynamically
    plt.imshow(img, cmap="gray")

print("Length of final_boxes[66]:", len(final_boxes[53]))
print("Contents of final_boxes[66]:", final_boxes[53])




# Loop through the final_boxes to make predictions using the FPR model
fpr_preds = []

for i in final_boxes:
    each_p = []
    for img in i:
        # Ensure the image has a shape of (50,50), resize if necessary
        if img.shape != (50, 50):
            img = cv2.resize(img, (50, 50))  # Ensure correct resizing
        img = img / 255.0  # Normalize the image
        img = np.reshape(img, (1, 50, 50, 1))  # Reshape for the model input
        pred = fpr_model.predict(img)  # Predict with the model
        pred = int(pred >= 0.5)  # Convert the probability to a binary class
        each_p.append(pred)
    fpr_preds.append(each_p)

# To check predictions for the 52nd image, adjust indexing here:
fpr_preds[53]  # Indexing is zero-based, so the 52nd image corresponds to index 51
bboxes[53]  # Also, this shows the bounding boxes corresponding to the 52nd image


# Adjusting the diameters by multiplying with the spacing factor (space[0])
for i in range(len(diams)):
    if len(diams[i]):
        for j in range(len(diams[i])):
            diams[i][j] = diams[i][j] * space[0]  # Multiply diameter by x-axis spacing

import pandas as pd
import copy
import cv2

# Create a DataFrame to store the details
df = pd.DataFrame(columns=['Layer', 'Position (x,y)', 'Diameter (mm)', 'BBox [(x1,y1),(x2,y2)]'])

final_img_bbox = []
cancer = []
e_lungs = copy.deepcopy(ct_norm_improved)  # Deepcopy of the lung images for modification

# Loop over the images, bboxes, predictions, centroids, and diameters
for i, (img, bbox, preds, cents, dms) in enumerate(zip(e_lungs, bboxes, fpr_preds, centroids, diams)):
    token = False  # Flag to track if cancerous regions were found
    img_copy = img.copy()  # Copy of the image for drawing bounding boxes

    # List to accumulate rows for the DataFrame
    rows = []

    # Loop through each bounding box, prediction, centroid, and diameter
    for box, pred, cent, dm in zip(bbox, preds, cents, dms):
        if pred:  # If the prediction indicates a cancerous region
            x1, y1 = box[0]  # Top-left corner
            x2, y2 = box[1]  # Bottom-right corner

            # Draw the bounding box on the image
            img_copy = cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255), 2)

            # Create a new row for the DataFrame with details about the detected region
            rows.append({
                'Layer': i,
                'Position (x,y)': f"{cent[::-1]}",  # Reversing centroids (to match (x, y) format)
                'Diameter (mm)': dm,
                'BBox [(x1,y1),(x2,y2)]': f"{list(box)}"
            })
            token = True  # Mark that a cancerous region was found

    # Append the modified image with bounding boxes to the list
    final_img_bbox.append(img_copy)

    # Add the new rows to the DataFrame if any cancerous regions were detected
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

    # Keep track of whether cancer was found in the image
    cancer.append(token)

# Display the DataFrame (optional)
print(df.head())


display2(final_img_bbox[40:55])
cancer[40:52]

df = df.reset_index(drop=True)
df.head()




# Define the folder path based on the FILE variable
folder = FILE.replace(".mhd", "")

# Create a results directory with the folder name inside the specified path
os.makedirs(f"C:/luna-16/Luna16/results/{folder}", exist_ok=True)

# Save the DataFrame `df` as a CSV in the new folder
df.to_csv(f"C:/luna-16/Luna16/results/{folder}/detections.csv", index=False)



# Set up the path for saving the video in the results folder
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
vid = cv2.VideoWriter(f"C:/luna-16/Luna16/results/{folder}/detections_2.mp4", fourcc, 5.0, (512, 512), False)

# Loop through each image in the final bounding box images
for i in range(len(final_img_bbox)):
    img = final_img_bbox[i].copy()
    img = cv2.putText(img, f"Layer: {i}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
    vid.write(img)

# Release the video writer
vid.release()







# Compute Dice coefficient for the entire test set
dice_scores = []
for i in range(len(testY)):
    dice = dice_coef(testY[i], pred[i])  # Compare ground truth (testY) with predicted (pred)
    dice_scores.append(K.eval(dice))  # Convert the tensor to a numpy value (using K.eval)

# Print the Dice scores for all images in the test set
print(f"Dice scores for all test images: {dice_scores}")

# Compute the average Dice score across the test set
average_dice = np.mean(dice_scores)
print(f"Average Dice coefficient: {average_dice}")




#to view all the data present to test 
import matplotlib.pyplot as plt

# Loop through each image in the test set
for i in range(len(testX)):
    # Get the test image and corresponding ground truth mask
    image = testX[i]  # The image from testX
    mask = testY[i]   # The ground truth mask from testY
    
    # Create a subplot with 1 row and 2 columns (one for the image and one for the mask)
    plt.figure(figsize=(10, 5))
    
    # Display the image
    plt.subplot(1, 2, 1)
    plt.imshow(image[..., 0], cmap='gray')  # Assuming image is a 3D array (H, W, C) and we use the first channel
    plt.title(f"Image {i + 1}")
    plt.axis('off')  # Turn off axis

    # Display the ground truth mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask[..., 0], cmap='gray')  # Assuming mask is also a 3D array (H, W, C)
    plt.title(f"Mask {i + 1}")
    plt.axis('off')  # Turn off axis

    # Show the plots
    plt.show()



#fpr model use case 
import os
import cv2
import copy
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

import SimpleITK as stk
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.cluster import KMeans
from skimage import measure

# Define the path for the dataset
PATH = "C:/luna-16/Luna16/subset0/subset0/"
FILE = "C:/luna-16/Luna16/subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.109002525524522225658609808059.mhd"

# Function to load .mhd files
def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    origin = np.array(list(mhdimage.GetOrigin()))
    space = np.array(list(mhdimage.GetSpacing()))
    return ct_scan, origin, space

# Custom evaluation metrics for the U-Net model
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

# Load and compile the model
model_path = "C:/luna-16/Luna16/models/FPR_classifier_model.h5"
custom_objects = {"dice_coef": dice_coef, "dice_coef_loss": dice_coef_loss}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Test loading .mhd file
ct_scan, origin, space = load_mhd(FILE)
print(f"CT Scan Shape: {ct_scan.shape}, Origin: {origin}, Spacing: {space}")