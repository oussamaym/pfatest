import cv2
import numpy as np
from tensorflow import keras
from train_model import label_dict 

# Load the pre-trained VGG16-based face recognition model
model = keras.models.load_model("fine_tuned_face_recognition_model.h5")

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

def predict_labels(model, face_images, label_dict):
    predictions = model.predict(np.array(face_images))
    predicted_labels = np.argmax(predictions, axis=1)
    label_names = [list(label_dict.keys())[list(label_dict.values()).index(label)] for label in predicted_labels]
    return label_names

while True:
    success, img_original = cap.read()
    faces = facedetect.detectMultiScale(img_original, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = img_original[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (224, 224))
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values to be between 0 and 1

        # Make a prediction using the face recognition model
        predicted_labels = predict_labels(model, [img], label_dict)

        if "Unknown" in predicted_labels:
            class_name = "Unknown"
            probability_value = 0.0  # Set probability to 0 for Unknown
        else:
            class_name = predicted_labels[0]
            probability_value = np.max(model.predict(img)) * 100  # Convert probability to percentage

        # Draw rectangle around the face and display class name and probability
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_original, str(class_name), (x, y - 10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_original, str(round(probability_value, 2)) + "%", (x, y + h + 25), font, 0.75, (255, 0, 0),
                    2, cv2.LINE_AA)

    cv2.imshow("Face Recognition", img_original)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()