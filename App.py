import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("Fire_detection_model.h5")  # Ensure the model was trained with 224x224 images

# Title and Subtitle (Centered with orange color)
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:orange;'>~ Fire Detection 🔥 ~</h1>
        <p style='font-size: 14px; color: gray;'>Made by Abdullah Haitham</p>
    </div>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    
    image = image.resize((244, 244))
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 3)

    # 🔍 Predict
    prediction = model.predict(image)[0][0]

    # ✅ Display result
    if prediction >= 0.5:
        label = "No Fire ✅"
        st.success(f"**{label}**")
        confidence = prediction
    else:
        label = "Fire 🔥"
        st.error(f"**{label}**")
        confidence = 1 - prediction

    st.info(f"**Confidence:** {confidence:.2%}")
