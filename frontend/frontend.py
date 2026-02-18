import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# CONFIG
# ==============================

API_BASE_URL = "https://heart-disease-api-uixi.onrender.com/predict"

MODEL_MAP = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "SVM": "svm"
}

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide"
)

# ==============================
# API FUNCTION
# ==============================

def call_prediction_api(model_name, payload):

    try:

        url = f"{API_BASE_URL}?model_name={model_name}"

        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()

        else:
            return {"error": f"API Error {response.status_code}"}

    except Exception as e:
        return {"error": str(e)}

# ==============================
# HEADER
# ==============================

st.title("❤️ Heart Disease Detection Dashboard")

st.write("Enter patient details and compare model predictions.")

st.divider()

# ==============================
# PATIENT INPUT SECTION
# ==============================

st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:

    age = st.number_input("Age", 29, 100, 54)

    sex = st.selectbox("Sex", [0, 1],
                       format_func=lambda x: "Female" if x == 0 else "Male")

    cp = st.selectbox(
        "Chest Pain Type",
        [0, 1, 2, 3],
        format_func=lambda x: [
            "Typical Angina",
            "Atypical Angina",
            "Non-anginal Pain",
            "Asymptomatic"
        ][x]
    )

    trestbps = st.slider("Resting BP", 80, 220, 130)

with col2:

    chol = st.slider("Cholesterol", 100, 600, 246)

    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

    restecg = st.selectbox("Rest ECG", [0, 1, 2])

    thalach = st.slider("Max Heart Rate", 60, 220, 150)

with col3:

    exang = st.selectbox("Exercise Induced Angina", [0, 1])

    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope", [1, 2, 3])

    ca = st.selectbox("Major Vessels", [0, 1, 2, 3])

    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

# ==============================
# MODEL SELECTION (SEPARATE)
# ==============================

st.divider()

st.subheader("Model Selection")

selected_model_display = st.selectbox(
    "Select Model for Primary Prediction",
    list(MODEL_MAP.keys())
)

# ==============================
# PREDICTION BUTTON (SEPARATE)
# ==============================

st.divider()

predict_button = st.button(
    "Predict Heart Disease Risk",
    use_container_width=True,
    type="primary"
)

# ==============================
# PREDICTION LOGIC
# ==============================

if predict_button:

    payload = {

        "age": age,
        "sex": sex,
        "chest_pain_type": cp,
        "resting_blood_pressure": trestbps,
        "cholesterol": chol,
        "fasting_blood_sugar": fbs,
        "resting_ecg": restecg,
        "max_heart_rate": thalach,
        "exercise_induced_angina": exang,
        "st_depression": oldpeak,
        "st_slope": slope,
        "num_major_vessels": ca,
        "thalassemia": thal

    }

    selected_model_key = MODEL_MAP[selected_model_display]

    st.divider()

    # Primary prediction
    st.subheader("Prediction Result")

    result = call_prediction_api(selected_model_key, payload)

    if "error" in result:

        st.error(result["error"])

    else:

        prediction = result["prediction"]
        probability = result["probability"]
        interpretation = result["interpretation"]

        col1, col2 = st.columns([2, 1])

        with col1:

            if prediction == 1:
                st.error(interpretation)
            else:
                st.success(interpretation)

        with col2:
            st.metric("Risk Probability", f"{probability:.2%}")

    # ==========================
    # MODEL COMPARISON GRAPH
    # ==========================

    st.divider()

    st.subheader("Model Comparison")

    comparison = []

    for model_display, model_key in MODEL_MAP.items():

        res = call_prediction_api(model_key, payload)

        if "error" not in res:

            comparison.append({
                "Model": model_display,
                "Probability": res["probability"]
            })

    df = pd.DataFrame(comparison)

    col1, col2 = st.columns([2, 1])

    with col1:

        fig, ax = plt.subplots()

        sns.barplot(data=df, x="Model", y="Probability", ax=ax)

        ax.set_title("Risk Comparison Between Models")

        st.pyplot(fig)

    with col2:

        st.dataframe(df, use_container_width=True)

# ==============================
# LINK TO METRICS PAGE
# ==============================

st.divider()

st.info("View detailed model performance metrics from the sidebar → Model Metrics")
