import os
import glob
import cv2
import numpy as np

# =============================================================================
# Dataset Loading
# =============================================================================
image_paths = glob.glob('Banana Disease Recognition Dataset/Augmented Images/*/*.jpg')
# =============================================================================
# 
# =============================================================================

preprocessed_images = []
labels = []
for i in image_paths:
    ll=i.split("\\")[1]
    img=cv2.imread(i)
    img=cv2.resize(img,(256,256))
    preprocessed_images.append(img)
    labels.append(ll)


class_names = sorted(os.listdir('Banana Disease Recognition Dataset/Augmented Images'))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
le = LabelEncoder()


Fused_features_encoded = le.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(preprocessed_images,Fused_features_encoded,test_size=0.2, random_state=42)


# Define base directories
base_dir = 'Banana Disease Recognition'
train_dir = os.path.join(base_dir, 'Train')
test_dir = os.path.join(base_dir, 'Test')

def create_folders(base_path, class_list):
    for cls in class_list:
        os.makedirs(os.path.join(base_path, cls), exist_ok=True)

create_folders(train_dir, class_names)
create_folders(test_dir, class_names)

# Function to save an image array
def save_image(image_array, dest_path):
    # Convert float to uint8 if necessary
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)
    if len(image_array.shape) == 2:
        cv2.imwrite(dest_path, image_array)
    elif len(image_array.shape) == 3:
        # BGR format for OpenCV
        cv2.imwrite(dest_path, image_array)

# Save training images
for idx, (img_array, label) in enumerate(zip(X_train, y_train)):
    class_folder = class_names[label]
    filename = f"train_{idx}.jpg"
    dest_path = os.path.join(train_dir, class_folder, filename)
    save_image(img_array, dest_path)

# Save testing images
for idx, (img_array, label) in enumerate(zip(X_test, y_test)):
    class_folder = class_names[label]
    filename = f"test_{idx}.jpg"
    dest_path = os.path.join(test_dir, class_folder, filename)
    save_image(img_array, dest_path)

