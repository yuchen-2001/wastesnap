# app.py

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import datetime
import uuid
import os

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
    image = image.convert("RGB") # ensure 3 channels
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
    region_rules = rules.get(region)
    if not region_rules:
        region_rules = rules.get("default", {})
    return region_rules.get(label) or rules["default"].get(label, "No recycling advice available.")


# Region dropdown with search field
region = st.selectbox("Select your region:", options=list(rules.keys()), index=0)



# Try an example image section
st.markdown("### Try an Example Image")

# available examples
example_images = {
    "Strawberry": "examples/strawberry.jpg",
    "Plastic Bottle": "examples/plasticbottle.jpg",
    "Carrots": "examples/carrots.jpg",
    "Curtain": "examples/curtain.jpg"
}

cols = st.columns(len(example_images))

selected_example = None

for idx, (label, path) in enumerate(example_images.items()):
    with cols[idx]:
        st.image(path, caption=label, use_container_width=True)
        if st.button(f"Try {label}"):
            selected_example = path


# Upload UI
uploaded_file = st.file_uploader("### Choose an image of waste", type=["jpg", "jpeg", "png"])

if uploaded_file is not None or selected_example is not None:
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
    else:
        img = Image.open(selected_example)
        st.image(img, caption=f"Example: {os.path.basename(selected_example)}", use_container_width=True)

    label, confidence = predict_image(img)
    tip = get_recycling_tip(label, region)

    label_display = "Recyclable" if label == "R" else "Organic Waste"
    st.markdown(f"### 🧾 Prediction: **{label_display}** ({confidence:.1%} confidence)")
    st.info(f"🧭 Recycling advice for **{region}**: {tip}")

    # Feedback section
    st.markdown("### 🙋 Was this prediction helpful?")
    col1, col2 = st.columns([1, 1])

    feedback_path = "feedback.json"  # or use full path if needed

    if col1.button("👍 Yes"):
        feedback = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "label": label,
            "region": region,
            "confidence": confidence,
            "feedback": "positive"
        }
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(feedback)
        with open(feedback_path, "w") as f:
            json.dump(logs, f, indent=2)
        st.success("✅ Thanks for your feedback!")

    if col2.button("👎 No"):
        feedback = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "label": label,
            "region": region,
            "confidence": confidence,
            "feedback": "negative"
        }
        if os.path.exists(feedback_path):
            with open(feedback_path, "r") as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(feedback)
        with open(feedback_path, "w") as f:
            json.dump(logs, f, indent=2)
        st.warning("Thanks — we'll use this to improve!")



    city_url = rules.get(region, {}).get("url")
    if city_url:
        st.markdown(f"[🔗 Official recycling guide for {region}]({city_url})")

    # Optional tips
    with st.expander("📚 Learn more about recycling guidelines"):
        if label == "R":
            st.markdown("""
            - ♻️ **Rinse containers** before recycling
            - 🧻 **No greasy paper/cardboard** in recycling
            - 🗞️ **Flatten boxes** to save space
            - ❌ **Avoid plastic bags** in curbside bins
            """)
        else:
            st.markdown("""
            - 🥬 **Compost food scraps** if possible
            - 🧻 **Soiled napkins and paper towels** go in compost
            - 🚫 **No plastics** in organic bins
            - 🌎 **Check local programs** for community compost drop-offs
            """)



