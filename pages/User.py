import streamlit as st
import pandas as pd
import pickle
import os
from data_simulator import get_combined_user_data, smoking_map
from chatbot_helper import initialize_chatbot, get_chat_response

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Helper function to load pickle files
def load_pickle(file_name):
    file_path = os.path.join(MODELS_DIR, file_name)
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load diabetes model components
diabetes_model = load_pickle('Diabetes_model.pkl')
diabetes_scaler = load_pickle('Diabetes_scaler.pkl')
diabetes_pca = load_pickle('Diabetes_pca.pkl')

# Load hypertension model components
hypertension_model = load_pickle('hypertension_model.pkl')
hypertension_scaler = load_pickle('hypertension_scaler.pkl')
hypertension_pca = load_pickle('hypertension_pca.pkl')

st.title("User Dashboard")

def predict_risks(full_df):
    """
    Predict diabetes and hypertension risk based on user data.

    Args:
        full_df (pd.DataFrame): User's full feature dataframe.

    Returns:
        pd.DataFrame: Dataframe with predictions.
    """
    full_df['smoking_history'] = full_df['smoking_history'].map(smoking_map)

    diab_features = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    diab_input = full_df[diab_features]
    diab_scaled = diabetes_scaler.transform(diab_input)
    diab_pca = diabetes_pca.transform(diab_scaled)
    full_df['diabetes'] = diabetes_model.predict(diab_pca)  # ✅ Changed to 'diabetes'

    hyper_features = ['gender', 'age', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']
    hyper_input = full_df[hyper_features]
    hyper_scaled = hypertension_scaler.transform(hyper_input)
    hyper_pca = hypertension_pca.transform(hyper_scaled)
    full_df['Hypertension Risk'] = hypertension_model.predict(hyper_pca)

    return full_df

# -------------------- User Health Data ------------------------
if 'dynamic_data' not in st.session_state:
    st.error("No data available yet. Please open Admin Dashboard first to initialize the data.")
else:
    combined_data = get_combined_user_data(st.session_state.dynamic_data)
    user1_data = combined_data[combined_data['User ID'] == 1].copy()
    user1_pred = predict_risks(user1_data)

    st.write("### Your Real-Time Health Data (User ID: 1)")
    st.table(user1_pred[['bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes', 'Hypertension Risk']])

    if user1_pred['diabetes'].values[0] == 1 or user1_pred['Hypertension Risk'].values[0] == 1:
        st.error("⚠ Warning: You are at risk. Please consult your doctor.")
    else:
        st.success("✅ You are currently stable.")


# -------------------- Chatbot Section ------------------------
st.write("---")
st.header("🩺 Health Assistant Chatbot")

# Initialize chatbot and memory (session scoped)
if 'chatbot' not in st.session_state:
    conversation_chain, memory = initialize_chatbot()
    st.session_state.chatbot = conversation_chain
    st.session_state.memory = memory

# Initialize chat messages list if not exists
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Display existing messages like ChatGPT style
for msg in st.session_state.chat_messages:
    st.write(f"🧑 **You:** {msg['user']}")
    st.write(f"🤖 **Assistant:** {msg['bot']}")

# Input inside a form to prevent continuous rerun
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Ask me about your health condition, risks, or general medical advice:", key="chat_input")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    response = get_chat_response(st.session_state.chatbot, user_input)
    st.session_state.chat_messages.append({"user": user_input, "bot": response})
