# 🍎🍊 Apple vs Orange Classifier

A simple web application for classifying images of apples and oranges using deep learning, built with Streamlit and TensorFlow.

## 🔗 Demo

Try the live demo: [appleorange-classifier.streamlit.app](https://appleorange-classifier.streamlit.app/)

## ✨ Features

* Upload images in PNG, JPG, or JPEG format and get instant predictions
* Choose between two models for comparison:

  * **Custom CNN** — built from scratch using 4 Conv2D layers with MaxPooling2D
  * **Transfer Learning (MobileNetV2)** — leverages a pretrained model for image classification
* Display confidence scores and probability comparison between Apple and Orange
* Prediction history maintained throughout the current session
* Dedicated information tabs explaining the models used

## 🛠️ Tech Stack

* [Streamlit](https://streamlit.io/) — web application framework
* [TensorFlow/Keras](https://www.tensorflow.org/) — deep learning framework
* **NumPy** & **Pillow** — image processing and numerical computation

## 📁 Project Structure

```text
Apple_Orange/
├── app.py                           # Main Streamlit application
├── custom_cnn_apple_orange.h5       # Custom CNN model
├── mobilenetv2_apple_orange.h5      # MobileNetV2 model
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## 🚀 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/rajabhibhawa/Apple_Orange.git
cd Apple_Orange
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

### 4. Open the application

Open your browser and go to:

```text
http://localhost:8501
```

## 🧠 How the Model Works

The uploaded image goes through the following preprocessing and classification pipeline:

1. The image is resized to **160 × 160 pixels**
2. Pixel values are normalized to a range of **0–1**
3. The preprocessed image is passed to the selected model
4. The model generates a probability score for the **Apple** and **Orange** classes

The models use a **binary classification output with a sigmoid activation function**. A score closer to **1** indicates **Orange**, while a score closer to **0** indicates **Apple**.

## 📊 Models

### Custom CNN

The Custom CNN model is designed and trained from scratch specifically for the Apple vs Orange classification task. It uses multiple convolutional and pooling layers to learn visual features such as shapes, textures, and patterns.

### MobileNetV2

The MobileNetV2 model uses **transfer learning** with pretrained ImageNet weights. This allows the model to leverage features learned from a large image dataset and adapt them to the Apple vs Orange classification task.

## 🎯 Project Goal

This project demonstrates how deep learning models can be deployed as an interactive web application for real-time image classification, while also providing a comparison between a **custom-built CNN** and a **transfer learning approach**.
