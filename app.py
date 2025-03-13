import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from io import BytesIO
import base64

# Function to Set Background Image
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{encoded_string}") no-repeat center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image
set_background("bac.jpg")  # Update with your image file

# Load Model Once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plant_disease_cnn_model.keras')

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    H, W, C = 224, 224, 3
    img = np.array(image)
    img = cv2.resize(img, (H, W))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, H, W, C)
    return img

# Function to predict disease
def model_predict(image):
    processed_img = preprocess_image(image)
    prediction = np.argmax(model.predict(processed_img), axis=-1)[0]
    return prediction

# Hide Sidebar Initially
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = False

def toggle_sidebar():
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible

# Main Page Title & Subtitle
st.markdown("<h1 style='text-align: center; color: black;'>🌿 Smart Plant Disease Prediction</h1>", unsafe_allow_html=True)

# Sidebar Toggle Button
if st.button("☰ Get Started"):
    toggle_sidebar()

# Conditional Sidebar Display
if st.session_state.sidebar_visible:
    with st.sidebar:
        st.title('Get Started')
        app_mode = st.radio('Select Page', ['Home', 'Disease Recognition'])
else:
    app_mode = "Home"

# Home Page
if app_mode == 'Home':
    st.markdown("<h2 style='text-align: center;color: darkgreen;'>Welcome to the Plant Disease Prediction System 🌱</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;color: black;'>Helps farmers and researchers detect plant diseases early.</p>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == 'Disease Recognition':
    #st.header("📸 Upload Image for Disease Detection")
    st.markdown("<h2 style='text-align: left;color: brown;'>Upload Image</h2>", unsafe_allow_html=True)
    # Image Upload
    test_image = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

    if test_image:
        # Display uploaded image
        image = Image.open(test_image)
        st.image(image, caption="Uploaded Image", width=200)  # Adjust width as needed

        # Convert to numpy format
        image = image.convert("RGB")

        # Predict Button
        if st.button("🔍 Predict Disease"):
            with st.spinner("Processing..."):
                try:
                    result_index = model_predict(image)
                    class_names = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy']
                    
                    # Display result
                    st.markdown(
                        f"""
                        <div style="
                        padding: 15px;
                        border-radius: 10px;
                        background-color: #DFF2BF;
                        color: #4F8A10;
                        font-size: 24px;
                        font-weight: bold;
                        text-align: center;
                        border: 2px solid #4F8A10;">
                        ✅ Model Prediction: {class_names[result_index]}
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")
