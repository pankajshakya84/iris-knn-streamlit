import streamlit as st
import numpy as np
import joblib

# Load trained KNN model and scaler
model = joblib.load("iris_knn_model.pkl")
scaler = joblib.load("iris_scaler.pkl")

st.set_page_config(
    page_title="Iris Flower Classification - KNN",
    page_icon="🌸",
    layout="centered"
)

st.title("🌸 Iris Flower Classification")
st.write("K-Nearest Neighbors (KNN) Machine Learning Model")

st.sidebar.header("Enter Flower Measurements")

sepal_length = st.sidebar.number_input(
    "Sepal Length (cm)", min_value=0.0, max_value=10.0,
    value=5.1, step=0.1
)
sepal_width = st.sidebar.number_input(
    "Sepal Width (cm)", min_value=0.0, max_value=10.0,
    value=3.5, step=0.1
)
petal_length = st.sidebar.number_input(
    "Petal Length (cm)", min_value=0.0, max_value=10.0,
    value=1.4, step=0.1
)
petal_width = st.sidebar.number_input(
    "Petal Width (cm)", min_value=0.0, max_value=10.0,
    value=0.2, step=0.1
)

if st.button("Predict Iris Flower"):
    input_data = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )

    # Apply the same scaling used during training
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    class_names = {
        0: "Iris Setosa",
        1: "Iris Versicolor",
        2: "Iris Virginica"
    }

    st.success(f"Prediction: **{class_names[prediction]}**")

    st.subheader("Prediction Probabilities")
    for class_id, probability in enumerate(probabilities):
        st.write(f"{class_names[class_id]}: {probability:.2%}")
        st.progress(float(probability))

st.markdown("---")
st.caption("Model: KNN | Dataset: Iris | Built with Python, Scikit-learn and Streamlit")
