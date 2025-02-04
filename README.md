# Image-Classification-Using-CNN

# Multiclass Fish Image Classification

## 📌 Project Overview

This project aims to classify fish images into multiple categories using deep learning techniques. It involves training a **CNN from scratch** and leveraging **transfer learning** with pre-trained models to enhance performance. Additionally, the project includes saving models for future use and deploying a **Streamlit application** to predict fish species from user-uploaded images.


## 🏢 Domain

**Image Classification**

## 🔍 Problem Statement

The goal is to develop a robust **deep learning model** to accurately classify fish species. The task includes:

- Training a **CNN model** from scratch.
- Experimenting with **five pre-trained models** (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0).
- Fine-tuning these models for better accuracy.
- Saving the **best-performing model**.
- Deploying the model as an **interactive web application** using Streamlit.

## 💼 Business Use Cases

- **Enhanced Accuracy:** Determine the best model for fish image classification.
- **Deployment Ready:** Develop a user-friendly web application for real-time predictions.
- **Model Comparison:** Evaluate and compare different models to select the most suitable approach.

## 🚀 Approach

### 📌 Data Preprocessing & Augmentation

- Rescale images to **[0, 1]** range.
- Apply **data augmentation** techniques (rotation, zoom, flipping) to enhance model robustness.

### 📌 Model Training

- Train a **CNN model from scratch**.
- Experiment with **five pre-trained models**.
- **Fine-tune** the models on the fish dataset.
- Save the **best-performing model** (`.h5` format) for future use.

### 📌 Model Evaluation

- Compare performance metrics: **Accuracy, Precision, Recall, F1-score, Confusion Matrix**.
- Visualize **training history** (Accuracy & Loss) for each model.

### 📌 Deployment

- Build a **Streamlit web application** to:
  - Allow users to **upload fish images**.
  - Predict and **display the fish category**.
  - Provide **confidence scores** for predictions.

## 🎯 Skills Gained

- Deep Learning
- Python & TensorFlow/Keras
- Streamlit for Model Deployment
- Data Preprocessing & Augmentation
- Transfer Learning
- Model Evaluation & Visualization

## 📂 Dataset

- The dataset consists of **fish images**, categorized into folders by species.
- Loaded using **TensorFlow's ImageDataGenerator** for efficient processing.
- **Format:** Zip file containing image folders.

## 🎯 Project Deliverables

- **Trained Models:** CNN and pre-trained models saved in `.h5` format.
- **Streamlit Application:** Interactive web app for real-time predictions.
- **Python Scripts:** For **training, evaluation, and deployment**.
- **Comparison Report:** Metrics & insights from all models.


