import numpy as np
import streamlit as st
import warnings

from modules.auth_manager import logout, setup_database
from modules.cookie_manager import get_cookie_manager


k = 8.617e-5  # Boltzmann constant in eV / K

warnings.filterwarnings("ignore")
st.set_page_config("TL Glow Curve Deconvolution", initial_sidebar_state="collapsed")
st.title("TL Glow Curve Deconvolution")

cm = get_cookie_manager()
if not cm.ready():
    st.error("Cookie manager is not ready. Please refresh the page.")
    st.stop()

if cm.get("init") is None:
    if not setup_database():
        st.error("Failed to setup database. Please refresh the page.")
        st.stop()
    cm["init"] = "true"
    cm.save()

is_logged_in = cm.get("username", "") != ""

if is_logged_in:
    cols = st.columns([4, 1])
    with cols[0]:
        st.text(f"Hello, {cm.get('username')}!")
    with cols[1]:
        if st.button("Log out"):
            st.info("Logging out...")
            logout()
            st.rerun()

    if st.session_state.get("uploaded", False):
        st.switch_page("pages/main.py")
    else:
        st.session_state.uploaded = False
        st.session_state.located_maxima = False
        st.session_state.n = 1
        st.session_state.b = [1.6]
        st.session_state.I_m = [0.0]
        st.session_state.T_m = [500.0]
        st.session_state.E = [1.0]
        st.session_state.scale_factor = 500.0
        st.session_state.method = "SLSQP"
        st.session_state.prop_coeff_tol = 0.1
        st.session_state.dd_ma_window_width = 15
else:
    cols = st.columns([3, 1, 1])
    with cols[1]:
        st.page_link("pages/register.py", label="Register")
    with cols[2]:
        st.page_link("pages/login.py", label="Log in")

cols = st.columns([3, 1])

with cols[0]:
    st.write("Upload the file containing the data (in csv format only). The file must not have any headers. The first column must contain the temperature values (in K), and the second column must contain the intensity values (in a.u.).")
    with st.form("upload"):
        file = st.file_uploader("Upload data file", "csv", help="You must be logged in to upload a file.", disabled=not is_logged_in)
        submitted = st.form_submit_button("Upload", help="You must be logged in to upload a file.", icon="🚀", disabled=not is_logged_in)
        if file and submitted:
        # if is_logged_in:
            content = file.getvalue()
            file.close()
            data = np.array([list(map(np.float64, row.split(","))) for row in content.decode("utf-8").splitlines()])
            # # with open("temp_file.csv", "r") as fp:
            #     reader = csv.reader(fp)
            #     data = np.array([list(map(np.float64, row)) for row in reader])
            st.session_state.T, st.session_state.intensity = data[:, 0], data[:, 1]
            st.session_state.uploaded = True
            st.switch_page("pages/main.py")
with cols[1]:
    st.image("images/csv_file_example.png", "Example of the data file", width="stretch")
