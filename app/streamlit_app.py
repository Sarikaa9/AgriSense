# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
from torchvision import transforms
from PIL import Image
import requests
import io

# =======================
# CONFIG / UTILS
# =======================
import config  # store weather_api_key here
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
from utils.model import ResNet9

# =======================
# LOAD MODELS
# =======================

# Plant disease model
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
                   'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                   'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                   'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                   'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
                   'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
                   'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
                   'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

# Crop recommendation model
crop_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_model_path, 'rb'))

# Fertilizer data
fertilizer_df = pd.read_csv('Data/fertilizer.csv')

# =======================
# FUNCTIONS
# =======================

def weather_fetch(city_name):
    """
    Fetch temperature and humidity of a city.
    Returns (temperature, humidity) or (None, None) if API fails.
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}appid={api_key}&q={city_name}"

    try:
        response = requests.get(complete_url)
        data = response.json()

        if "main" in data:
            temperature = round(data["main"]["temp"] - 273.15, 2)  # Kelvin → Celsius
            humidity = data["main"]["humidity"]
            return temperature, humidity
        else:
            print("Weather API error:", data)
            return None, None
    except Exception as e:
        print("Error fetching weather:", e)
        return None, None

def predict_image(img, model=disease_model):
    """Predict disease from image"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction

# =======================
# STREAMLIT APP
# =======================

st.set_page_config(page_title="Harvestify", layout="wide")
st.title("🌱 Harvestify - Smart Farming Assistant")

menu = ["Home", "Crop Recommendation", "Fertilizer Suggestion", "Disease Detection"]
choice = st.sidebar.selectbox("Navigation", menu)

# =======================
# HOME PAGE
# =======================
if choice == "Home":
    st.subheader("Welcome to Harvestify! 🌾")
    st.write("Choose an option from the sidebar to get started.")

# =======================
# CROP RECOMMENDATION
# =======================
elif choice == "Crop Recommendation":
    st.subheader("Crop Recommendation")

    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Pottasium (K)", min_value=0, max_value=200, value=50)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    city = st.text_input("City Name", value="Delhi")

    if st.button("Predict Crop"):
        temp, hum = weather_fetch(city)
        if temp is None or hum is None:
            st.error("⚠️ Could not fetch weather data. Please check the city name or your API key.")
        else:
            st.write(f"Temperature: {temp}°C")
            st.write(f"Humidity: {hum}%")
            features = np.array([[N, P, K, temp, hum, ph, rainfall]])
            predicted_crop = crop_recommendation_model.predict(features)[0]
            st.success(f"Recommended Crop: {predicted_crop}")

# =======================
# FERTILIZER SUGGESTION
# =======================
elif choice == "Fertilizer Suggestion":
    st.subheader("Fertilizer Suggestion")

    crop_name = st.selectbox("Select Crop", fertilizer_df['Crop'].unique())
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
    P = st.number_input("Phosphorous (P)", min_value=0, max_value=200, value=50)
    K = st.number_input("Pottasium (K)", min_value=0, max_value=200, value=50)

    if st.button("Suggest Fertilizer"):
        df = fertilizer_df
        nr = df[df['Crop'] == crop_name]['N'].iloc[0]
        pr = df[df['Crop'] == crop_name]['P'].iloc[0]
        kr = df[df['Crop'] == crop_name]['K'].iloc[0]

        n = nr - N
        p = pr - P
        k = kr - K
        temp_dict = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_val = temp_dict[max(temp_dict.keys())]

        if max_val == "N":
            key = 'NHigh' if n < 0 else 'Nlow'
        elif max_val == "P":
            key = 'PHigh' if p < 0 else 'Plow'
        else:
            key = 'KHigh' if k < 0 else 'Klow'

        fertilizer_text = fertilizer_dic[key]
        fertilizer_text = fertilizer_text.replace("<br/>", "\n").replace("<i>", "").replace("</i>", "")
        st.markdown(f"**Fertilizer Suggestion:**\n{fertilizer_text}")


# =======================
# DISEASE DETECTION
# =======================
# =======================
# DISEASE DETECTION
# =======================
elif choice == "Disease Detection":
    st.subheader("Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload a leaf image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        try:
            img_bytes = uploaded_file.read()
            prediction = predict_image(img_bytes)

            # CLEAN HTML TAGS
            disease_text = disease_dic[prediction]
            for tag in ["<br/>", "<b>", "</b>", "<i>", "</i>"]:
                disease_text = disease_text.replace(tag, "")

            st.markdown(f"**Disease Detected:**\n{disease_text}")

            st.image(Image.open(io.BytesIO(img_bytes)), caption="Uploaded Leaf", use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

