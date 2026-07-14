from nlp.summary import generate_summary
import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MediLens AI",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load Models
# -----------------------------
rf_model = joblib.load("models/random_forest.pkl")
scaler = joblib.load("models/scaler.pkl")
cnn_model = tf.keras.models.load_model("models/mobilenet.keras")

# -----------------------------
# Sidebar
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🩺 Diabetes Prediction",
        "🫁 Chest X-Ray",
        "ℹ About"
    ]
)

# ==================================================
# HOME
# ==================================================

if page == "🏠 Home":

    st.title("🩺 MediLens AI")

    st.subheader("AI Diagnostic & Insight System")

    st.markdown("---")

    st.write("""
This project combines

- Machine Learning
- Deep Learning
- Computer Vision
- Explainable AI
- Medical Report Generation

into one healthcare application.
""")

# ==================================================
# DIABETES
# ==================================================

elif page == "🩺 Diabetes Prediction":

    st.title("🩺 Diabetes Risk Prediction")
    st.markdown("Enter the patient's clinical information below.")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies",0,20)
        glucose = st.number_input("Glucose",0,250)
        bp = st.number_input("Blood Pressure",0,150)
        skin = st.number_input("Skin Thickness",0,100)

    with col2:
        insulin = st.number_input("Insulin",0,900)
        bmi = st.number_input("BMI",0.0,70.0)
        pedigree = st.number_input("Diabetes Pedigree Function",0.0,3.0)
        age = st.number_input("Age",1,120)

    if st.button("🔍 Predict Diabetes"):

        values = np.array([[

            pregnancies,
            glucose,
            bp,
            skin,
            insulin,
            bmi,
            pedigree,
            age

        ]])

        scaled = scaler.transform(values)

        prediction = rf_model.predict(scaled)[0]

        probability = rf_model.predict_proba(scaled)[0][1]

        st.divider()

        c1,c2,c3 = st.columns(3)

        with c1:
            st.metric("Risk", "HIGH" if prediction else "LOW")

        with c2:
            st.metric("Probability", f"{probability*100:.2f}%")

        with c3:
            if prediction:
                st.metric("Status","⚠️ Attention")
            else:
                st.metric("Status","✅ Healthy")

        if prediction:

            st.error("Patient is at HIGH risk of diabetes.")

        else:

            st.success("Patient is at LOW risk of diabetes.")

        st.subheader("🤖 AI Medical Summary")

        if prediction:

            st.info(f"""
The AI model predicts an increased likelihood of diabetes.

Main indicators:

• Elevated glucose level

• BMI = {bmi}

• Age = {age}

This prediction should be confirmed through laboratory tests and medical consultation.
""")

        else:

            st.info("""
The patient's values indicate a relatively low risk of diabetes.

Maintain a healthy lifestyle and continue routine health checkups.
""")

# ==================================================
# X-RAY
# ==================================================

elif page == "🫁 Chest X-Ray":

    st.title("🫁 Chest X-Ray Classification")

    uploaded_file = st.file_uploader(
        "Upload Chest X-Ray",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded X-Ray", use_container_width=True)

        img = image.resize((224,224))
        img = np.array(img, dtype=np.float32)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = cnn_model.predict(img, verbose=0)

        confidence = float(prediction[0][0])

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        with col2:

            if confidence > 0.5:
                st.metric("Prediction", "Pneumonia")
            else:
                st.metric("Prediction", "Normal")

        if confidence > 0.5:

            st.error("""
Possible Pneumonia detected.

Please consult a healthcare professional.
""")

        else:

            st.success("""
No signs of Pneumonia detected.

This AI prediction should not replace a medical diagnosis.
""")
# ==================================================
# ABOUT
# ==================================================

elif page == "ℹ About":

    st.title("About MediLens AI")

    st.write("""

Major Project

Technologies Used

• Python

• Streamlit

• Scikit-Learn

• TensorFlow

• MobileNetV2

• Pandas

• NumPy

• OpenCV

• SHAP

• HuggingFace

""")