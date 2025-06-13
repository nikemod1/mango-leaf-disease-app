import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/mango_leaf_model.h5")
    return model

model = load_model()

# Class labels (must match training order)
CLASS_NAMES = ['Anthracnose', 'Bacterial Canker', 'Healthy']

# App title
st.title("🍃 Mango Leaf Disease Detector")
st.write("Upload a mango leaf image, and this app will predict the disease using a CNN model.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction with loading spinner
    with st.spinner("🔍 Analyzing the leaf..."):
        prediction = model.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    # Show result
    st.success(f"🌿 Predicted Disease: **{predicted_class}**")
    st.info(f"🔢 Confidence: {confidence:.2f}%")
