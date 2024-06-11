import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time
from keras.models import load_model

# Load the trained Keras model
model = load_model('Model/keras_model.h5')

bg = None
folder = "Data/yo"

# Dictionary mapping label numbers to actual label text
labels = {
    0: "1", 1: "2", 2: "3", 3: "4", 4: "5",
    5: "fist", 6: "ok", 7: "vdown", 8: "vup", 9: "yo"
}

def run_avg(image, accumWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

# Function to preprocess image for the model
def preprocess_image(image):
    # Resize image to match model's expected sizing
    image = cv2.resize(image, (224, 224))  # Resize to (224, 224)
    # Convert to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Normalize pixel values to be between 0 and 1
    image = image / 255.0
    # Expand dimensions to match the shape required by the model
    image = np.expand_dims(image, axis=0)
    return image

if __name__ == "__main__":
    accumWeight = 0.5
    camera = cv2.VideoCapture(0)
    top, right, bottom, left = 10, 350, 225, 590
    num_frames = 0
    calibrated = False
    counter = 0

    while(True):
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)
        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                print("[STATUS] please wait! calibrating...")
            elif num_frames == 29:
                print("[STATUS] calibration successful...")
        else:
            hand = segment(gray)
            if hand is not None:
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow("Thesholded", thresholded)

                # Preprocess the segmented hand image for the model
                preprocessed_image = preprocess_image(thresholded)

                # Predict the sign using the model
                predictions = model.predict(preprocessed_image)
                predicted_label = np.argmax(predictions)
                sign_label = labels[predicted_label]
                print("Predicted Sign Label:", sign_label)
                
                # Display the sign label text on the frame
                cv2.putText(clone, "Sign: " + sign_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', thresholded)
            print(counter)

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
