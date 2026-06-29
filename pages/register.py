import streamlit as st

from modules.auth_manager import load_user, register
from modules.cookie_manager import get_cookie_manager


st.set_page_config("Register", initial_sidebar_state="collapsed")

cm = get_cookie_manager()
if not cm.ready():
    st.error("Cookie manager is not ready. Please refresh the page.")
    st.rerun()

if cm.get("username", "") != "":
    st.switch_page("home.py")

st.header("Register")
with st.form("register_form"):
    username = st.text_input("Username", placeholder="Choose a username")
    password = st.text_input("Password", type="password", placeholder="Choose a password")
    confirm_password = st.text_input("Confirm password", type="password", placeholder="Confirm your password")
    if st.form_submit_button("Register"):
        if not username or not password:
            st.error("Username and password cannot be empty.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            st.info("Registering...")
            if not register(username, password):
                st.error("Username already exists.")
            else:
                if load_user(username, password):
                    st.success("Registration successful!")
                    st.switch_page("home.py")
                else:
                    st.error("Failed to create user.")

st.page_link("pages/login.py", label="Already have an account? Log in here.")
st.page_link("home.py", label="Back to Home", icon="🏠")
