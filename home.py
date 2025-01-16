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

# def logout():
#     cookies["user_id"] = ""
#     cookies["username"] = ""
#     if "engine" in st.session_state:
#         st.session_state.engine.dispose(False)
#     st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)
#     st.switch_page("")

# warnings.filterwarnings("ignore")

cookies = initCookies()

# Ensure that the cookies are ready
if not cookies.ready():
    st.error("Cookies not initialised yet.")
    st.stop()

# st.title("TL Glow Curve Deconvolution")

if cookies.get("user_id", "") == "":
    # Not logged in
    st.title("TL Glow Curve Deconvolution")

    st.page_link("./pages/login.py")
    st.page_link("./pages/register.py")
    # st.page_link("./pages/view_tables.py")
    # st.page_link("./pages/create_tables.py")
    # st.page_link("./pages/clear_tables.py")

    # if "engine" not in st.session_state:
    #     st.session_state.engine = create_engine("sqlite+pysqlite:///pages/common/databases/server_side.db", echo=True)
    # sql = """
    # CREATE TABLE IF NOT EXISTS users_credentials (
    #     username VARCHAR(256),
    #     password_hash VARCHAR(256),
    #     user_id VARCHAR(256) PRIMARY KEY
    # )
    # """
    # executeSQL(sql, st.session_state.engine, True)
else:
    # Logged in
    st.switch_page("./pages/main.py")
    # col_1, col_2 = st.columns([4, 1])
    # with col_1:
    #     st.title("TL Glow Curve Deconvolution")
    # with col_2:
    #     st.button("Logout", on_click=logout)

    # if "engine" not in st.session_state:
    #     st.session_state.engine = create_engine(f"sqlite+pysqlite:///pages/common/databases/users/db_{cookies["user_id"]}.db", echo=True)
