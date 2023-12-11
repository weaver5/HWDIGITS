import os
import numpy as np
import cv2
import tensorflow as tf
import time
from picamera import PiCamera
from time import sleep
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sense_hat import SenseHat

model_filepath = '/home/pi/Downloads/MNIST_Michaela_Model1.h5'

# My model has already been trained, just loading it in using Keras
model = tf.keras.models.load_model(model_filepath, compile=False)

def imgprocessed(image):
    # Preprocess the image (resize, normalize, etc.)
    imgprocessed = cv2.resize(image, (28, 28))  # Adjust the size as needed
    imgprocessed = imgprocessed.reshape((1, 28, 28, 1)).astype('float32') / 255.0

    # Record the start time for prediction
    start_time = time.time()

    # Make predictions using the loaded model
    predic = model.predict(imgprocessed)

    # Record the end time for prediction
    end_time = time.time()

    # Get the predicted digit and confidence level
    pred_val = np.argmax(predic)
    confidence = predic[0][pred_val]

    # Print the predicted digit and confidence level with modified text
    print(f"MY Digit: {pred_val}, Confidence/Acc: {confidence}")

    # Calculate and print the prediction time
    prediction_time = end_time - start_time
    print(f"Prediction Time: {prediction_time} seconds")

    return pred_val

# Function to display the digit on the Sense HAT in red text on a green background
def display_sensehat(sense, digit):
    # Define colors
    red = (255, 0, 0)    # RGB values for red
    green = (0, 255, 0)  # RGB values for green

    # Display the digit on the Sense HAT in red text on a green background
    sense.show_message(str(digit), text_colour=red, back_colour=green)

# Open a video capture object
cap = cv2.VideoCapture(0)

# Initialize the Sense HAT
sense = SenseHat()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Invert the frame
    inverted_frame = cv2.bitwise_not(frame)

    # Convert the inverted frame to grayscale
    gray = cv2.cvtColor(inverted_frame, cv2.COLOR_BGR2GRAY)

    # Adjust the threshold value to make it less aggressive
    _, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Display the result
    cv2.imshow('White Digit on Black Paper Effect', thresholded)

    # Check for the 'c' key to capture the image
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # Process the captured image and make predictions
        predicted_digit = imgprocessed(thresholded)

        # Display the predicted digit on the Sense HAT in red text on a green background
        display_sensehat(sense, predicted_digit)

    # Update the Sense HAT display to keep the digit static
    elif key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
