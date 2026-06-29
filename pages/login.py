import streamlit as st

from modules.auth_manager import load_user
from modules.cookie_manager import get_cookie_manager


st.set_page_config("Login", initial_sidebar_state="collapsed")

cm = get_cookie_manager()
if not cm.ready():
    st.error("Cookie manager is not ready. Please refresh the page.")
    st.rerun()

if cm.get("username", "") != "":
    st.switch_page("home.py")

st.header("Login")
with st.form("login_form"):
    username = st.text_input("Username", placeholder="Enter your username")
    password = st.text_input("Password", type="password", placeholder="Enter your password")
    if st.form_submit_button("Log in"):
        if not username or not password:
            st.error("Username and password cannot be empty.")
        elif load_user(username, password):
            st.success("Logging in...")
            st.switch_page("home.py")
        else:
            st.error("Incorrect username or password.")

st.page_link("pages/register.py", label="Don’t have an account? Register here.")
st.page_link("home.py", label="Back to Home", icon="🏠")
