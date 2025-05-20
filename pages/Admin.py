import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
from streamlit_autorefresh import st_autorefresh
from data_simulator import generate_dynamic_data, get_combined_user_data, smoking_map
from file_extractor import extract_text_from_file
from report_generator import generate_lab_report_summary
from pdf_generator import generate_pdf_report

models_dir = Path('models')

# Load diabetes model components
diabetes_model = pickle.load(open(models_dir / 'diabetes_model.pkl', 'rb'))
diabetes_scaler = pickle.load(open(models_dir / 'diabetes_scaler.pkl', 'rb'))
diabetes_pca = pickle.load(open(models_dir / 'diabetes_pca.pkl', 'rb'))

# Load hypertension model components
hypertension_model = pickle.load(open(models_dir / 'hypertension_model.pkl', 'rb'))
hypertension_scaler = pickle.load(open(models_dir / 'hypertension_scaler.pkl', 'rb'))
hypertension_pca = pickle.load(open(models_dir / 'hypertension_pca.pkl', 'rb'))

st.title("Admin Dashboard")

def predict_risks(full_df):
    full_df['smoking_history'] = full_df['smoking_history'].map(smoking_map)

    diab_features = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    diab_input = full_df[diab_features]
    diab_scaled = diabetes_scaler.transform(diab_input)
    diab_pca = diabetes_pca.transform(diab_scaled)
    full_df['diabetes'] = diabetes_model.predict(diab_pca)

    hyper_features = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    hyper_input = full_df[hyper_features]
    hyper_scaled = hypertension_scaler.transform(hyper_input)
    hyper_pca = hypertension_pca.transform(hyper_scaled)
    full_df['Hypertension Risk'] = hypertension_model.predict(hyper_pca)

    return full_df

# -------------------- Auto-refresh every 20 seconds ------------------------
st_autorefresh(interval=20 * 1000, key="data_refresh")

# -------------------- Dynamic Data Simulation ------------------------
if 'dynamic_data' not in st.session_state:
    st.session_state.dynamic_data = generate_dynamic_data()

# Update data on each refresh
st.session_state.dynamic_data = generate_dynamic_data()
combined_data = get_combined_user_data(st.session_state.dynamic_data)
combined_data = predict_risks(combined_data)

st.write("### Live All Users Data (Admin)")
st.table(combined_data[['User ID', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes', 'Hypertension Risk']])

# Show alerts for high-risk users
high_risk_users = combined_data[(combined_data['diabetes'] == 1) | (combined_data['Hypertension Risk'] == 1)]
if not high_risk_users.empty:
    st.error("⚠ Users at Risk Detected! Please notify doctors.")
    st.table(high_risk_users[['User ID', 'diabetes', 'Hypertension Risk']])
else:
    st.success("✅ All users currently stable.")

st.write("---")

# -------------------- Lab Report Upload & Doctor Summary ------------------------
st.header("🧪 Upload Lab Test Report for Doctor Summary")

uploaded_file = st.file_uploader("Upload patient's lab test report", type=['pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg'])

if uploaded_file:
    with st.spinner("Extracting and processing report..."):
        text = extract_text_from_file(uploaded_file, filename=uploaded_file.name)

    if not text.strip():
        st.error("❌ Could not extract any meaningful text from the uploaded report.")
    else:
        with st.spinner("Generating doctor summary report..."):
            summary = generate_lab_report_summary(text)
            st.write("### Doctor Summary Report:")
            st.write(summary)
            
            pdf_bytes = generate_pdf_report(summary)

            # Download button
            st.download_button(
                label="📄 Download Doctor Report as PDF",
                data=pdf_bytes,
                file_name="doctor_report.pdf",
                mime="application/pdf"
            )
