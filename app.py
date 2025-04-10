import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import json

# Load model and labels
model = load_model("model/waste_classifier.h5")
class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Load regional recycling rules
with open("recycling_rules.json", "r") as f:
    rules = json.load(f)

def predict_image(img: Image.Image):
    img = img.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]
    class_idx = np.argmax(prediction)
    confidence = float(prediction[class_idx])
    return class_labels[class_idx], confidence

def get_recycling_tip(label, region="default"):
    region_rules = rules.get(region, rules["default"])
    return region_rules.get(label, "No guidance available.")

# Streamlit app
st.set_page_config(page_title="WasteSnap", layout="centered")
st.title("♻️ WasteSnap")
st.write("Upload an image of your waste item to find out if it's recyclable!")

# Region selector
region = st.selectbox("Select your region:", options=list(rules.keys()), index=0)

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict_image(img)
    tip = get_recycling_tip(label, region)

    st.markdown(f"### 🧾 Prediction: **{label.capitalize()}** ({confidence:.1%} confidence)")
    st.info(f"🧭 Recycling advice for **{region}**: {tip}")
