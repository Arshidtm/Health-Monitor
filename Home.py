import streamlit as st

# Simulated user database (For prototype only)
USER_CREDENTIALS = {
    "user": "user123",
    "admin": "admin123"
}

# Title
st.title("AI Health Monitoring System (Prototype)")

# Sidebar login form
st.sidebar.header("Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if st.sidebar.button("Login"):
    if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
        st.success(f"Logged in as {username.capitalize()}")
        if username == "user":
            st.info("Go to the 'User' page from the left menu.")
        elif username == "admin":
            st.info("Go to the 'Admin' page from the left menu.")
    else:
        st.error("Invalid username or password.")

st.write("Use the left sidebar to navigate to User or Admin dashboard after logging in.")
