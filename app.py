# app.py

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json

# Set up the Streamlit page
st.set_page_config(page_title="WasteSnap", layout="centered")
st.title("♻️ WasteSnap")
st.write("Upload an image of your waste item to find out if it's recyclable!")

# Define waste class labels (should match model training order)
#class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_labels = ['O', 'R']  # 0 = Organic, 1 = Recyclable

# Load regional recycling rules
@st.cache_data
def load_rules():
    with open("recycling_rules.json", "r") as f:
        return json.load(f)

rules = load_rules()

# Load the TensorFlow Lite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="model/waste_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def predict_image(image):
    image = image.resize((224, 224))  # Or match your model's expected size exactly
    img_array = np.array(image) / 255.0

    # Ensure shape is [1, height, width, channels]
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Get input tensor details (make sure shape and dtype match)
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    if img_array.shape != tuple(input_shape):
        st.error(f"Expected input shape {input_shape}, but got {img_array.shape}")
        return "Error", 0.0
    if img_array.dtype != input_dtype:
        img_array = img_array.astype(input_dtype)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx = np.argmax(output_data)
    confidence = float(output_data[class_idx])
    return class_labels[class_idx], confidence


# Get tip for a region + label
def get_recycling_tip(label, region="default"):
    region_rules = rules.get(region, rules["default"])
    return region_rules.get(label, "No guidance available.")

# Region dropdown
region = st.selectbox("Select your region:", options=list(rules.keys()), index=0)

# Upload UI
uploaded_file = st.file_uploader("Choose an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(img)
    tip = get_recycling_tip(label, region)

    st.markdown(f"### 🧾 Prediction: **{label.capitalize()}** ({confidence:.1%} confidence)")
    st.info(f"🧭 Recycling advice for **{region}**: {tip}")
