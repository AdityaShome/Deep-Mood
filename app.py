import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('emotion_model_mobilenetv2.h5')

index_to_class = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

st.title("Emotion Detection from Image")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    emotion = index_to_class[predicted_class]

    st.subheader(f'Prediction: {emotion.upper()}')
    #st.text(f'Confidence: {confidence*100:.2f}%')

