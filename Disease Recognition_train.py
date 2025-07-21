# =======================================
# 1. Import Required Libraries
# =======================================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# =======================================
# 2. Set Paths and Parameters
# =======================================
train_path = 'Banana Disease Recognition/Train'
test_path = 'Banana Disease Recognition/Test'
img_size = (128, 128)
batch_size = 32

# =======================================
# 3. Data Preprocessing
# =======================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# =======================================
# 4. Build CNN Model
# =======================================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])


# =======================================
# 5. Compile the Model
# =======================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# =======================================
# 6. Train the Model
# =======================================
epochs=100
# history = model.fit(train_data,
#                     validation_data=test_data,
#                     epochs=epochs)
# # Save the entire model
# model.save("model/Banana Disease Recognition_model.h5")
# np.save('model/Banana Disease Recognition_history.npy', history.history)


# =======================================
# 7. Plot Accuracy and Loss
# =======================================
plt.figure(figsize=(12, 5))

loaded_history = np.load('model/Banana Disease Recognition_history.npy', allow_pickle=True).item()
font = {'family' : 'Times New Roman',
        'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)
plt.figure(figsize=(6, 4))
plt.plot(loaded_history['loss'], label="Training Loss", color='blue')
plt.plot(loaded_history['val_loss'], label="Testing Loss", color='orange')
# plt.ylim(0,1)
plt.xlim(0,100)
plt.xlabel("Epochs",fontsize=18,weight="bold",fontname='Times New Roman')
plt.ylabel("Loss",fontsize=18,weight="bold",fontname='Times New Roman')
plt.legend(loc='upper right', fontsize=14)
plt.xticks(fontsize=16,weight="bold",fontname='Times New Roman')
plt.yticks(fontsize=16,weight="bold",fontname='Times New Roman')
plt.grid(True,linestyle="--",alpha=0.7)
plt.tight_layout()
plt.show()
plt.savefig("Result/Banana Disease Recognition_Loss.jpg", dpi=800)

# Plot Accuracy
plt.figure(figsize=(6, 4))
plt.plot(loaded_history['accuracy'], label="Training Accuracy", color='blue')
plt.plot(loaded_history['val_accuracy'], label="Testing Accuracy", color='orange')
# plt.ylim(0,1)
plt.xlim(0,100)
plt.xlabel("Epochs",fontsize=18,weight="bold",fontname='Times New Roman')
plt.ylabel("Accuracy",fontsize=18,weight="bold",fontname='Times New Roman')
plt.legend(loc='lower right', fontsize=14)
plt.xticks(fontsize=16,weight="bold",fontname='Times New Roman')
plt.yticks(fontsize=16,weight="bold",fontname='Times New Roman')
plt.grid(True,linestyle="--",alpha=0.7)
plt.tight_layout()
plt.show()
plt.savefig("Result/Banana Disease Recognition_Accuracy.jpg", dpi=800)


# =======================================
# 8. Predict on a New Image
# =======================================

# To load the full model
from tensorflow.keras.models import load_model
model = load_model("model/Banana Disease Recognition_model.h5")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_name = list(train_data.class_indices.keys())[class_index]
    print(f"Predicted Class: {class_name}")

predict_image('Banana Disease Recognition/Test/Banana Bract Mosaic Virus Disease/test_6.jpg')

# =======================================
# 9. Evaluate Model: Confusion Matrix and Metrics
# =======================================
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Get all test data (images and labels)
test_images, test_labels = [], []
test_steps = len(test_data)

for _ in range(test_steps):
    img_batch, label_batch = next(test_data)
    test_images.extend(img_batch)
    test_labels.extend(label_batch)

test_images = np.array(test_images)
test_labels = np.array(test_labels)

# Get true class indices
true_classes = np.argmax(test_labels, axis=1)

# Predict classes
pred_probs = model.predict(test_images)
pred_classes = np.argmax(pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(true_classes, pred_classes)
class_names = list(test_data.class_indices.keys())


import seaborn as sn
import pandas as pd
columns =  ['Black Sigatoka',
 'Bract Mosaic Virus',
 'Healthy','Insect Pest','Moko','Panama','Yellow Sigatoka']
df_cm = pd.DataFrame(cm, columns, columns)
plt.figure(figsize=(7.5, 6))
sn.heatmap(df_cm, annot=True,fmt="d",cmap="plasma", annot_kws={"size": 14,"fontname":"Times New Roman","weight":"bold"}) # font size
plt.xticks(fontsize=16,weight="bold",fontname='Times New Roman')
plt.yticks(fontsize=16,weight="bold",fontname='Times New Roman')
plt.xlabel("Predicted Label",fontsize=18,weight="bold",fontname='Times New Roman')
plt.ylabel("Actual Label",fontsize=18,weight="bold",fontname='Times New Roman')
plt.tight_layout()
plt.show()
plt.savefig("Result/Banana Disease Recognition_Confusion_Matrix.jpg", dpi=800)



# Classification Report (includes Precision, Recall, F1-score per class)
report = classification_report(true_classes, pred_classes, target_names=class_names)
print("Classification Report:\n", report)

from sklearn.metrics import  roc_auc_score
acc = accuracy_score(true_classes, pred_classes)
prec = precision_score(true_classes, pred_classes, average='macro')
rec = recall_score(true_classes, pred_classes, average='macro')
f1 = f1_score(true_classes, pred_classes, average='macro')
auc = roc_auc_score(test_labels, pred_probs, multi_class='ovr')
    

print(f"Overall Accuracy   : {acc:.4f}")
print(f"Overall Precision  : {prec:.4f}")
print(f"Overall Recall     : {rec:.4f}")
print(f"Overall F1-score   : {f1:.4f}")
print(f"Overall AUC-ROC    : {auc:.4f}")