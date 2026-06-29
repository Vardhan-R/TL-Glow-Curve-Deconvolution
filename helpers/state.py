import numpy as np
import streamlit as st

from helpers.tl_model import estimate_im_from_tm, find_double_derivative_minima


def update_n() -> None:
    n = int(st.session_state.n_input)
    diff = n - len(st.session_state.b)
    if diff > 0:
        st.session_state.b.extend([1.6] * diff)
        st.session_state.I_m.extend([50000.0] * diff)
        st.session_state.T_m.extend([500.0] * diff)
        st.session_state.E.extend([np.random.normal(1, 0.1)] * diff)
    elif diff < 0:
        st.session_state.b = st.session_state.b[:n]
        st.session_state.I_m = st.session_state.I_m[:n]
        st.session_state.T_m = st.session_state.T_m[:n]
        st.session_state.E = st.session_state.E[:n]
    st.session_state.n = n
    st.session_state.I_m = [estimate_im_from_tm(tm_value, st.session_state.T, st.session_state.intensity) for tm_value in st.session_state.T_m]


def sync_tm_sliders() -> None:
    st.session_state.T_m = [st.session_state[f"Tm_slider_{i}"] for i in range(st.session_state.n)]
    st.session_state.I_m = [estimate_im_from_tm(tm_value, st.session_state.T, st.session_state.intensity) for tm_value in st.session_state.T_m]


def initialise_from_data() -> None:
    seed_tm, _ = find_double_derivative_minima(
        st.session_state.T,
        st.session_state.intensity,
        st.session_state.dd_ma_window_width,
    )
    st.session_state.n = len(seed_tm)
    st.session_state.b = [1.6] * st.session_state.n
    st.session_state.I_m = [estimate_im_from_tm(tm_value, st.session_state.T, st.session_state.intensity) for tm_value in st.session_state.T_m]
    st.session_state.T_m = seed_tm.copy()
    st.session_state.E = np.random.normal(1, 0.1, st.session_state.n).tolist()
    st.session_state.scale_factor = 500.0
    st.session_state.method = "SLSQP"
    st.session_state.prop_coeff_tol = 0.1
    st.session_state.located_maxima = True
