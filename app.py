
from pathlib import Path

import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "clothing_classifier.joblib"

st.set_page_config(
    page_title="AI Outfit Planner",
    page_icon="👗",
    layout="centered",
)

st.title("👗 AI Outfit Planner")
st.write(
    "Upload a clothing image and get outfit ideas "
    "based on weather, occasion, and personal style."
)


def extract_features(image):
    image = image.convert("RGB").resize((64, 64))
    pixels = np.asarray(image, dtype=np.float32) / 255.0

    gray = (
        0.299 * pixels[:, :, 0]
        + 0.587 * pixels[:, :, 1]
        + 0.114 * pixels[:, :, 2]
    )

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True,
    )

    histograms = [
        np.histogram(
            pixels[:, :, channel],
            bins=16,
            range=(0, 1),
            density=True,
        )[0]
        for channel in range(3)
    ]

    return np.concatenate([hog_features, *histograms]).reshape(1, -1)


def suggest_outfit(category, weather, occasion, style):
    combinations = {
        "top": "Try jeans, tailored trousers, or a skirt.",
        "pants": "Try pairing them with a fitted top, shirt, or sweater.",
        "dress": "Try adding a light jacket, flats, or sneakers.",
        "skirt": "Try a tucked-in top, shirt, or cardigan.",
        "shoes": "Try pairing them with jeans and a top, a dress, or trousers.",
        "outwear": "Try layering over a basic top with jeans or tailored trousers.",
    }

    advice = [combinations.get(category, "Try pairing this item with complementary basics.")]

    if weather == "Hot":
        advice.append("Choose lightweight, breathable fabrics.")
    elif weather == "Cold":
        advice.append("Consider warm layers and a suitable outer layer.")
    elif weather == "Rainy":
        advice.append("Consider water-resistant footwear and a rain-friendly outer layer.")
    else:
        advice.append("Use comfortable layers suited to mild weather.")

    if occasion == "Formal":
        advice.append("Choose clean lines and polished accessories.")
    elif occasion == "Party":
        advice.append("Add a statement accessory to make the outfit stand out.")
    elif occasion == "College":
        advice.append("Prioritize comfort and easy-to-move-in pieces.")
    else:
        advice.append("Keep the combination comfortable and easy to style.")

    advice.append(f"Style direction: {style}.")
    return advice


if not MODEL_PATH.exists():
    st.error(
        "Trained model not found. Run train_model.py first "
        "and make sure models/clothing_classifier.joblib exists."
    )
    st.stop()


@st.cache_resource
def load_model(path):
    return joblib.load(path)


try:
    model = load_model(str(MODEL_PATH))
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a clothing image",
    type=["jpg", "jpeg", "png", "webp"],
)

weather = st.selectbox("Weather", ["Mild", "Hot", "Cold", "Rainy"])
occasion = st.selectbox("Occasion", ["Casual", "College", "Formal", "Party"])
style = st.selectbox("Preferred style", ["Minimal", "Streetwear", "Classic", "Trendy"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded clothing item", use_container_width=True)

        features = extract_features(image)
        prediction = str(model.predict(features)[0]).lower()

        st.success(f"Predicted clothing category: {prediction.title()}")

        st.subheader("Outfit suggestions")
        for suggestion in suggest_outfit(prediction, weather, occasion, style):
            st.write(f"- {suggestion}")

        st.caption(
            "Suggestions are rule-based ideas, not predictions from a trained "
            "outfit compatibility model."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
