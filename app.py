import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Set up Streamlit
st.title("SIGN LANGUAGE ALPHABET RECOGNIZER")
st.write("This app uses a trained model to classify hand gestures in real-time")
st.image("venv/sign language image.png", use_column_width=True)

labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","HI","NAMASTHE"]

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/K NAGA SAI/PycharmProjects/pythonProject2/Model/keras_model.h5", "C:/Users/K NAGA SAI/PycharmProjects/pythonProject2/Model/labels.txt")

# Create an output element to display video stream
output = st.empty()

# Create Start and Stop buttons
start_button = st.button("Start", key="start")

stop_button = st.button("Stop", key="stop")

video_started = False
last_time = time.time()
captured_alphabets = []

while True:
    if start_button:
        video_started = True

    if stop_button:
        video_started = False

    if video_started:
        success, img = cap.read()

        if img is not None:
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((300, 300, 3), np.uint8) * 255
                imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]

                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = 300 / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, 300))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((300 - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                else:
                    k = 300 / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (300, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((300 - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

                cv2.rectangle(imgOutput, (x - 20, y - 70),
                              (x + 70, y - 20), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x + 10, y - 45), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - 20, y - 20),
                              (x + w + 20, y + h + 20), (255, 0, 255), 4)

                # Measure the time interval and capture the alphabet every 2 seconds
                current_time = time.time()
                if current_time - last_time >= 2:
                    captured_alphabets.append(labels[index])
                    last_time = current_time

        # Display the video stream with annotations in the Streamlit app
        output.image(imgOutput, channels="BGR")
    else:
        # Clear the output to stop displaying the video
        output = st.empty()

# Release the webcam when done
cap.release()