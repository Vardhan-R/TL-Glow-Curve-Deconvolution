from hashlib import sha256
from pages.common.cookies_manager import initCookies
from pages.common.databases_manager import executeSQL
from scipy.optimize import minimize
from sqlalchemy import create_engine
from streamlit_cookies_manager import EncryptedCookieManager
import altair as alt
import csv
import numpy as np
import os
import pandas as pd
import streamlit as st
import time
import warnings

def dispMenu():
    col_new, col_open = st.columns([1, 1])
    with col_new:
        st.button("New analysis")
    with col_open:
        st.button("Open analysis")

def logout():
    cookies["user_id"] = ""
    cookies["username"] = ""
    cookies.save()
    if "engine" in st.session_state:
        st.session_state.engine.dispose(False)
    st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)
    st.switch_page("./home.py")

warnings.filterwarnings("ignore")

cookies = initCookies()

# Ensure that the cookies are ready
if not cookies.ready():
    st.error("Cookies not initialised yet.")
    st.stop()

if cookies.get("user_id", "") == "":
    # Not logged in
    st.switch_page("./home.py")
else:
    # Logged in
    col_1, col_2 = st.columns([4, 1])
    with col_1:
        st.title("TL Glow Curve Deconvolution")
    with col_2:
        st.button("Logout", on_click=logout)

    if "engine" not in st.session_state:
        st.session_state.engine = create_engine(f"sqlite+pysqlite:///pages/common/databases/users/db_{cookies["user_id"]}.db", echo=True)

    dispMenu()
