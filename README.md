# 🖐️ Sign Language Recognition Using Deep Learning

This project implements a **Sign Language Recognition System** using **Convolutional Neural Networks (CNNs)**. The model is trained to recognize American Sign Language (ASL) alphabet gestures from images and perform real-time classification using a webcam.

---

## 🚀 Features
- **Dataset Preprocessing**: Augments and prepares training, validation, and test datasets.
- **CNN Model Training**: Implements a deep learning model using TensorFlow/Keras.
- **Real-Time Prediction**: Uses OpenCV for real-time gesture recognition.
- **Alphabet Classification**: Recognizes ASL alphabets (`A-Z`), including special gestures (`del`, `nothing`, `space`).

---

## 📁 Project Structure
```
📂 Sign Language Recognition
├── 📄 preprocessdataset.py   # Prepares dataset and trains the model
├── 📄 model.py               # Loads trained model and performs real-time prediction
├── 📂 asl_alphabet_train     # Training dataset directory (not included in repo)
├── 📂 asl_alphabet_test      # Test dataset directory (not included in repo)
└── 📄 README.md              # Project documentation
```

---

## 📊 Dataset
- The model is trained on the **ASL Alphabet Dataset**, containing labeled images for each sign.
- Data augmentation techniques (rotation, width/height shifts, shear, zoom) are applied to improve robustness.

---

## 🛠️ Setup & Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

### 2️⃣ Install Dependencies
```sh
pip install numpy tensorflow opencv-python
```

### 3️⃣ Prepare Dataset
Place your dataset inside the appropriate directories:
```
📂 asl_alphabet_train/
📂 asl_alphabet_test/
```

### 4️⃣ Train the Model
Run the preprocessing and training script:
```sh
python preprocessdataset.py
```

### 5️⃣ Run Real-Time Prediction
Once the model is trained, test real-time sign detection using:
```sh
python model.py
```

---

## 🧠 Model Architecture
The CNN model consists of:
- **3 Convolutional Layers** with ReLU activation
- **Max-Pooling Layers** to reduce spatial dimensions
- **Flatten & Fully Connected Layers** with a dropout layer
- **Softmax Output Layer** for multi-class classification

---

## 📌 How It Works
1. The camera captures a frame.
2. The image is **preprocessed** and **resized** to `(64x64)`.
3. The trained CNN model predicts the **most likely ASL sign**.
4. The predicted label is **displayed on the screen**.

---

## 🎯 Future Improvements
- Support for **dynamic gestures** using RNN/LSTM.
- Improve accuracy with a **larger dataset**.
- Develop a **web/app interface** for better accessibility.

---

## 🤝 Contributing
Feel free to **fork** this repository and submit a pull request for improvements!

---

## 📬 Contact
- **GitHub:** [@ishaansxna](https://github.com/ishaansxna)  
- **LinkedIn:** [Ishaan Saxena](https://linkedin.com/in/ishaan-saxena-21942b277)  
- **Email:** [ishaansaxena2022@gmail.com](mailto:ishaansaxena2022@gmail.com)  

---

📝 **Note**: This project is built for educational purposes and can be improved further with advanced models.
```
