import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import os
import time

# Set Streamlit page title and icon
st.set_page_config(page_title="Fish Classification App", page_icon="🐟", layout="wide")

# Create a menu with sidebar
menu = st.sidebar.radio("Navigation", ["Home", "Fish Classification"])

if menu == "Home":
    # Project Overview for Home Page with Animation
    st.markdown("""
        # Fish Classification App 🐟 
        **An AI-powered tool to classify fish species with deep learning.**
        
        ### 🔹 Key Features
        ✅ **Deep Learning Model** – Trained using CNN & fine-tuned pre-trained models (VGG16, ResNet50, etc.).  
        ✅ **Streamlit App** – User-friendly web app for fish classification.  
        ✅ **Image Upload & Prediction** – Users can upload fish images and get real-time predictions.  
        ✅ **Confidence Score** – Displays model confidence for transparency.  
        ✅ **Deployment** – Hosted on **Hugging Face Spaces** for easy access.  

        ### 🔹 Tech Stack
        📌 **TensorFlow / Keras** – Model training & inference  
        📌 **Streamlit** – Web-based user interface  
        📌 **Hugging Face Spaces** – Model deployment  
        📌 **OpenCV & NumPy** – Image processing  

        ### 🔹 Goal
        
                To provide an **automated fish classification tool** for fisheries, researchers, and hobbyists to quickly and accurately identify fish species.  
        ---
    """)

elif menu == "Fish Classification":
    st.title("Fish Classification 🐟")
    
    # Animated loading effect
    with st.spinner("Loading best model information..."):
        time.sleep(2)
    
    # Load best model metadata
    if not os.path.exists("best_model_info.json"):
        st.error("⚠️ Best model info file not found. Please run your model training script first.")
        st.stop()
    
    with open("best_model_info.json", "r") as f:
        best_model_data = json.load(f)
    
    best_model_name = best_model_data.get("best_model_name", "Unknown Model")
    best_model_accuracy = best_model_data.get("best_model_accuracy", 0.0)
    model_path = f"{best_model_name}.h5"
    
    # Load the best model with animation
    with st.spinner(f"Loading Model: {best_model_name}..."):
        if not os.path.exists(model_path):
            st.error(f"⚠️ Best model file '{model_path}' not found. Please ensure the model is saved correctly.")
            st.stop()
        model = tf.keras.models.load_model(model_path)
        time.sleep(2)
    
    # Define class names (Ensure this matches your training labels)
    class_names = [
        "Animal Fish", "Bass", "Black Sea Sprat", "Gilt Head Bream", "Horse Mackerel", 
        "Red Mullet", "Red Sea Bream", "Sea Bass", "Shrimp", "Striped Red Mullet", "Trout"
    ]
    
    st.write(f"### 📌 Using Best Model: **{best_model_name}** (Accuracy: **{best_model_accuracy:.2f}%**)")
    st.write("Upload an image of a fish, and the model will classify it!")
    
    # Upload image
    uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Load and preprocess image with animation
        with st.spinner("Processing image..."):
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Convert to batch format
            img_array /= 255.0  # Normalize
            time.sleep(2)
        
        # Make prediction with animation
        with st.spinner("Classifying fish species..."):
            predictions = model.predict(img_array)
            predicted_class = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100  # Convert to percentage
            time.sleep(2)
    
        # Display result
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        st.write(f"### 🐠 Predicted Species: **{predicted_class}**")
        st.write(f"### 🔥 Confidence: **{confidence:.2f}%**")
        
        # Add a confidence bar
        st.progress(int(confidence))
        
        st.markdown("---")
        st.success("✅ Prediction completed successfully!")
    
# Footer
st.markdown("""
    **📌 Note:** This model classifies fish species based on trained data. 
    If incorrect classifications occur, please ensure clear fish images. 
    
    📖 [https://github.com/RajeshwariArumugam/Image-Classification-Using-CNN.git](#) | 🤖 Model by AI/ML Developer
""")
