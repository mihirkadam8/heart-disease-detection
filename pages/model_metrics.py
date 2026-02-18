import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Model Metrics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Model Performance Metrics")



data = {

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],

    "Accuracy": [0.85, 0.82, 0.89, 0.87],
    "Precision": [0.84, 0.80, 0.90, 0.86],
    "Recall": [0.86, 0.83, 0.88, 0.87],
    "F1 Score": [0.85, 0.81, 0.89, 0.86]

}

df = pd.DataFrame(data)

st.dataframe(df, use_container_width=True)

st.subheader("Graphical Comparison")

fig, ax = plt.subplots()

df.set_index("Model").plot(kind="bar", ax=ax)

st.pyplot(fig)

if st.button("⬅ Back to Dashboard", use_container_width=True):
    st.switch_page("frontend.py")
