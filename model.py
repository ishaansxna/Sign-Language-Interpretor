import numpy as np
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import cv2

# Disabling the oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Defining the image size used during training
img_size = (64, 64) 

# Defining class labels 
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

# Loading the trained model
try:
    model = load_model('asl_alphabet_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Real-time prediction function
def real_time_prediction(model, img_size, class_labels):
    cap = cv2.VideoCapture(0)  # Initialize the camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, img_size)  
        normalized_frame = resized_frame / 255.0  
        input_frame = np.expand_dims(normalized_frame, axis=0)  

        # Prediction the class
        try:
            predictions = model.predict(input_frame)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_label = class_labels[predicted_class[0]]
        except Exception as e:
            print(f"Error during prediction: {e}")
            predicted_label = "Error"

        # Displaying of the prediction on the frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Interpreter", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run real-time prediction
real_time_prediction(model, img_size, class_labels)