import pandas as pd
import random

# Fixed user data
fixed_user_data = pd.DataFrame([
    {"User ID": 1, "gender": 1, "age": 45, "heart_disease": 1, "smoking_history": "current"},
    {"User ID": 2, "gender": 0, "age": 50, "heart_disease": 0, "smoking_history": "never"},
    {"User ID": 3, "gender": 1, "age": 35, "heart_disease": 0, "smoking_history": "former"},
    {"User ID": 4, "gender": 0, "age": 60, "heart_disease": 1, "smoking_history": "never"},
    {"User ID": 5, "gender": 1, "age": 40, "heart_disease": 0, "smoking_history": "current"}
])

# Smoking category encoding
smoking_map = {
    'No Info': 0,
    'never': 1,
    'former': 2,
    'current': 3,
    'not current': 4,
    'ever': 5
}

# Simulate dynamic readings (shared)
def generate_dynamic_data():
    data = []
    for user_id in range(1, 6):
        data.append({
            "User ID": user_id,
            "bmi": round(random.uniform(18.5, 35), 1),
            "HbA1c_level": round(random.uniform(5.5, 9.0), 1),
            "blood_glucose_level": random.randint(80, 250)
        })
    return pd.DataFrame(data)

def get_combined_user_data(dynamic_data):
    return pd.merge(fixed_user_data, dynamic_data, on="User ID")
