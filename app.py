import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
MODEL_PATH = "skin_disease_model.h5"
IMAGE_SIZE = (224, 224)

# Class labels and corresponding disease names
CLASS_INFO = {
    'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}
CLASS_LABELS = list(CLASS_INFO.keys())

# Load the model
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file '{MODEL_PATH}' not found!")
        return None
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# App title and description
st.title("🩺 Skin Disease Classification App")
st.markdown("""
Upload a skin lesion image, and the model will predict the skin disease class and provide additional details.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing
        img = image.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        if model:
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_label = CLASS_LABELS[predicted_index]
            predicted_disease = CLASS_INFO[predicted_label]
            confidence = float(np.max(prediction))

            # Output
            st.success(f"✅ **Predicted Class:** `{predicted_label.upper()}`")
            st.info(f"🧬 **Disease:** {predicted_disease}")
            st.info(f"📊 **Confidence:** {confidence * 100:.2f}%")
        else:
            st.warning("⚠️ Model not loaded properly.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
