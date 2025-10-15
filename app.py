import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Configuration ---
st.set_page_config(
    page_title="Breast Cancer Prognosis Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Feature Names ---
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se',
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
    'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# --- Default Values ---
FEATURE_DEFAULTS = {
    'radius_mean': (14.5, 6.9, 28.1, 0.1),
    'texture_mean': (19.2, 9.7, 39.3, 0.1),
    'perimeter_mean': (92.0, 43.7, 188.5, 0.1),
    'area_mean': (654.9, 143.5, 2501.0, 1.0),
    'smoothness_mean': (0.096, 0.052, 0.163, 0.001),
    'compactness_mean': (0.104, 0.019, 0.345, 0.001),
    'concavity_mean': (0.088, 0.0, 0.427, 0.001),
    'concave points_mean': (0.048, 0.0, 0.201, 0.001),
    'symmetry_mean': (0.181, 0.106, 0.304, 0.001),
    'fractal_dimension_mean': (0.062, 0.050, 0.097, 0.0001),

    'radius_se': (0.4, 0.1, 2.0, 0.001),
    'texture_se': (1.2, 0.3, 4.9, 0.001),
    'perimeter_se': (2.8, 0.8, 22.0, 0.1),
    'area_se': (40.0, 6.8, 542.2, 0.1),
    'smoothness_se': (0.007, 0.001, 0.031, 0.0001),
    'compactness_se': (0.025, 0.002, 0.135, 0.0001),
    'concavity_se': (0.031, 0.0, 0.396, 0.0001),
    'concave points_se': (0.011, 0.0, 0.053, 0.0001),
    'symmetry_se': (0.020, 0.008, 0.078, 0.0001),
    'fractal_dimension_se': (0.003, 0.001, 0.030, 0.0001),

    'radius_worst': (17.5, 7.9, 36.0, 0.1),
    'texture_worst': (25.6, 12.0, 49.5, 0.1),
    'perimeter_worst': (115.0, 50.4, 251.2, 0.1),
    'area_worst': (880.0, 185.2, 4254.0, 1.0),
    'smoothness_worst': (0.132, 0.071, 0.223, 0.001),
    'compactness_worst': (0.254, 0.027, 1.058, 0.001),
    'concavity_worst': (0.272, 0.0, 1.252, 0.001),
    'concave points_worst': (0.114, 0.0, 0.291, 0.001),
    'symmetry_worst': (0.323, 0.156, 0.664, 0.001),
    'fractal_dimension_worst': (0.080, 0.055, 0.207, 0.0001),

    'id': (1234567, 1, 99999999, 1),
}

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pickled model from the file system."""
    try:
        with open('best_LR1.pkl', 'rb') as f:  # ✅ updated filename
            model = pickle.load(f)
        st.success("✅ Model 'best_LR1.pkl' loaded successfully.")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'best_LR1.pkl' not found in the application directory. Please ensure it is present.")
        return "MODEL_NOT_FOUND"

model = load_model()

# --- Prediction ---
def predict_cancer_real(model_obj, input_data):
    """Performs real prediction using the loaded model."""
    if model_obj == "MODEL_NOT_FOUND":
        return 'B'

    df = pd.DataFrame([input_data])
    features_only = df.drop(columns=['id']).reindex(columns=FEATURE_NAMES)

    try:
        prediction = model_obj.predict(features_only)
        result = 'M' if prediction[0] == 1 else 'B'
        return result
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return 'B'

# --- Helper Functions ---
def format_name(name):
    return name.replace('_', ' ').title()

def create_input_widgets():
    """Creates all 32 input widgets with unique keys."""
    st.sidebar.title("Patient ID")
    id_default, id_min, id_max, id_step = FEATURE_DEFAULTS['id']
    patient_id = st.sidebar.number_input(
        "Patient ID (for tracking)",
        min_value=id_min,
        max_value=id_max,
        value=id_default,
        step=id_step
    )

    st.header("Tumor Measurements")
    st.markdown("Enter the 31 measurements below to predict if the tumor is **Benign (B)** or **Malignant (M)**.")

    input_data = {'id': patient_id}
    cols = st.columns(3)

    for i, feature in enumerate(FEATURE_NAMES):
        col = cols[i % 3]
        default_val, min_val, max_val, step_val = FEATURE_DEFAULTS.get(feature, (1.0, 0.0, 10.0, 0.001))
        input_data[feature] = col.number_input(
            format_name(feature),
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            step=step_val,
            format="%f" if step_val < 0.1 else "%.1f",
            key=f"{feature}_{i}"  # ✅ unique key for each widget
        )
    return input_data

def display_prediction(prediction):
    """Displays final prediction nicely."""
    st.divider()
    st.subheader("Prediction Result")

    if prediction == 'M':
        st.error("## Malignant (M)")
        st.write("The model predicts a **malignant (M)** tumor. This suggests the presence of cancer.")
        st.warning("🚨 **Consult a Medical Professional:** This is an AI-based prediction and must be verified by a doctor.")
    else:
        st.success("## Benign (B)")
        st.write("The model predicts a **benign (B)** tumor. This suggests the tumor is likely non-cancerous.")
        st.info("✅ **Consult a Medical Professional:** This is an AI-based prediction and should not replace medical advice.")

# --- Main App ---
def main():
    st.title("Breast Cancer Prognosis Predictor 🔬")
    st.caption("Powered by Logistic Regression Model (`bes_LR.pkl`)")

    input_data = create_input_widgets()

    st.sidebar.divider()
    if st.sidebar.button("Predict Prognosis", type="primary"):
        prediction = predict_cancer_real(model, input_data)
        display_prediction(prediction)

if __name__ == "__main__":
    main()
