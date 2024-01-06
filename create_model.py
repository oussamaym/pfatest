import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow import layers

# Function to load and resize images from a folder
def load_images_from_folder(folder, target_size=(128, 128), num_images=501):
    images = []
    for filename in os.listdir(folder)[:num_images]:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  
        if img is not None:
            img = cv2.resize(img, target_size) 
            images.append(img)
    return images
# Function to create a face recognition model
def create_face_recognition_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Path to the root folder containing subfolders for each person
root_folder = "images"  # Update this if needed
num_images_per_person = 501  # You can adjust this based on your dataset

# Dictionary to map subfolder names to numerical labels
label_dict = {folder: label for label, folder in enumerate(os.listdir(root_folder))}
print(label_dict)

# Lists to store images and corresponding labels
data = []
labels = []

# Load, resize, and append images to the data list
for person_folder in os.listdir(root_folder):
    person_path = os.path.join(root_folder, person_folder)
    images = load_images_from_folder(person_path, num_images=num_images_per_person)
    data.extend(images)
    labels.extend([label_dict[person_folder]] * len(images))

# Convert lists to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize pixel values to be between 0 and 1
data = data / 255.0

# Reshape data for input to CNN (add channel dimension)
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 3)  # Adjust to handle color images

print(np.unique(labels))


# Check if a pre-trained model exists
model_file = "face_recognition_model.h5"
if os.path.exists(model_file):
    # Load the existing model
    face_model = keras.models.load_model(model_file)
else:
    # Create and compile the face recognition model if no existing model is found
    input_shape = (data.shape[1], data.shape[2], 3)  # Adjust to handle color images
    num_classes = len(label_dict)
    face_model = create_face_recognition_model(input_shape, num_classes)
# Print a sample of labels and filenames before training
# Print a sample of labels and filenames before training



# Train the model with more epochs and increased complexity
face_model.fit(data, labels, epochs=10, verbose=1)
 # Adjust the number of epochs based on your needs

# Save the trained model for future use
face_model.save(model_file)
