import cv2
from keras.models import load_model
import numpy as np

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

# Load the pre-trained face recognition model
model = load_model("face_recognition_model.h5")
def get_class_name(class_no):
    # Adjust this function based on your class nqames
    if class_no == 0:
        return "Andrew"
    elif class_no == 1:
        return "Tristan"

while True:
    success, img_original = cap.read()
    faces = facedetect.detectMultiScale(img_original, 1.3, 5)

    for x, y, w, h in faces:
        crop_img = img_original[y:y + h, x:x + w]
        img = cv2.resize(crop_img, (128, 128))
        img = img / 255.0  # Normalize pixel values to be between 0 and 1
        img = img.reshape(1, 128, 128, 3)  # Adjust to handle color images

        # Make a prediction using the face recognition model
        prediction = model.predict(img)

        # Get the predicted class indexq
        class_index = np.argmax(prediction, axis=-1)

        # Get the probability valueq
        probability_value = np.amax(prediction)

        # Draw rectangle around the face and display class name and probability
        cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_original, str(get_class_name(class_index[0])), (x, y - 10), font, 0.75, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(img_original, str(round(probability_value * 100, 2)) + "%", (x, y + h + 25), font, 0.75, (255, 0, 0),
                    2, cv2.LINE_AA)

    cv2.imshow("Face Recognition", img_original)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
