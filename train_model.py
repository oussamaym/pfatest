import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16

def load_images_from_folder(folder, target_size=(224, 224), num_images=501):
    images = []
    for filename in os.listdir(folder)[:num_images]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.resize(img, target_size)
            images.append(img)
    return images

root_folder = "images"
num_images_per_person = 501

label_dict = {folder: label for label, folder in enumerate(os.listdir(root_folder))}
print(label_dict)

data = []
labels = []

for person_folder in os.listdir(root_folder):
    person_path = os.path.join(root_folder, person_folder)
    images = load_images_from_folder(person_path, num_images=num_images_per_person)
    data.extend(images)
    labels.extend([label_dict[person_folder]] * len(images))

data = np.array(data)
labels = np.array(labels)

data = data / 255.0


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

# Build a new model on top of VGG16
model = keras.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()


model.fit(data, labels, epochs=2, verbose=1)


model.save("fine_tuned_face_recognition_model.h5")
