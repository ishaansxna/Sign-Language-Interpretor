import numpy as np
import os
import tensorflow as tf
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import cv2

# Defining paths
train_dir = 'C:\\Users\\Ishaan Saxena\\Desktop\\SIGN LANGUAGE\\asl_alphabet_train'  # Path to training dataset
test_dir = 'C:\\Users\\Ishaan Saxena\\Desktop\\SIGN LANGUAGE\\asl_alphabet_test'    # Path to test dataset
img_size = (64, 64) 
batch_size = 32

# Data augmentation and preprocessing for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,  
    rotation_range=10,  
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    validation_split=0.2  # Split 20% of training data for validation
)

# Loading training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Use 80% of training data for training
)

# Loading validation data
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Use 20% of training data for validation
)

# Loading test data (no data augmentation, only rescaling)
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Do not shuffle test data
)

# Checking class indices
print("Class indices:", train_generator.class_indices)

# Defining class labels (including 'del', 'nothing', 'space')
class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    'del', 'nothing', 'space'
]

# Defining the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)), 
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Preventing the overfitting
    Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# Compiling of the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Printing of the model summary
model.summary()

# Training the model
epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Evaluate the model on the test set
if test_generator.samples > 0:
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
else:
    print("Error: No test images found.")

# Saving the model
model.save('asl_alphabet_model.h5')
print("Model saved successfully.")

# Real-time prediction
def real_time_prediction(model, img_size, class_labels):
    cap = cv2.VideoCapture(0)  # Initialize the camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Preprocess the frame
        resized_frame = cv2.resize(frame, img_size)
        normalized_frame = resized_frame / 255.0
        input_frame = np.expand_dims(normalized_frame, axis=0)

        # Predict the class
        predictions = model.predict(input_frame)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]

        # Display the prediction
        cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Language Interpreter", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Load the trained model for real-time interpretation
try:
    model = load_model('asl_alphabet_model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Run real-time prediction
real_time_prediction(model, img_size, class_labels)