import altair as alt
import pandas as pd
import streamlit as st

from helpers.plotting import plotLineGraph as plot_line_graph, plotScatterGraph as plot_scatter_graph
from helpers.state import initialise_from_data as initialise_from_data_helper, sync_tm_sliders as sync_tm_sliders_helper, update_n as update_n_helper
from helpers.tl_model import estimate_im_from_tm as estimate_im_from_tm_helper, execute as execute_helper, fom as fom_helper, get_double_derivative_curve as get_double_derivative_curve_helper, multi_tl_equation as multi_tl_equation_helper, tl_equation as tl_equation_helper, half_max_temperatures
from modules.auth_manager import logout
from modules.cookie_manager import get_cookie_manager


st.set_page_config("Your Workspace", initial_sidebar_state="collapsed")

if st.session_state.get("uploaded") is None:
    st.switch_page("home.py")

cm = get_cookie_manager()

# Ensure that the cookies are ready
if not cm.ready():
    st.error("Cookies not initialised yet. Please refresh the page.")
    st.stop()

if cm.get("username", "") == "":
    # Not logged in
    st.switch_page("home.py")

# Logged in
st.set_page_config(f"{cm.get('username')}’s Workspace", initial_sidebar_state="collapsed")
st.header(f"{cm.get('username')}’s Workspace")

cols = st.columns([5, 3, 2])
with cols[1]:
    if st.button("Back to file upload"):
        st.session_state.uploaded = False
        st.switch_page("home.py")
with cols[2]:
    if st.button("Log out"):
        st.info("Logging out...")
        logout()
        st.switch_page("home.py")

if st.session_state.get("located_maxima") is None:
    initialise_from_data_helper()

st.number_input("n (number of peaks)", 1, value=st.session_state.n, key="n_input", on_change=update_n_helper)
st.session_state.dd_ma_window_width = st.number_input(
    "Double derivative MA window width",
    min_value=1,
    value=int(st.session_state.dd_ma_window_width),
    step=1,
    help="Moving average window width applied to the double derivative before locating zero crossings.",
)

dd_x, dd_y, dd_minima_x, dd_minima_y = get_double_derivative_curve_helper(
    st.session_state.T,
    st.session_state.intensity,
    st.session_state.dd_ma_window_width,
)

tm_cols = st.columns(2)
for i in range(st.session_state.n):
    with tm_cols[i % 2]:
        st.slider(
            f"T_m[{i + 1}]",
            float(st.session_state.T.min()),
            float(st.session_state.T.max()),
            value=float(st.session_state.T_m[i]),
            key=f"Tm_slider_{i}",
            on_change=sync_tm_sliders_helper,
        )

intensity_line = plot_line_graph(st.session_state.T, st.session_state.intensity, "#17becf", "Temperature (K)", "Intensity (au)")
tm_scatter = plot_scatter_graph(st.session_state.T_m, st.session_state.I_m, "#d62728", "Temperature (K)", "Intensity (au)")
st.write(intensity_line + tm_scatter)

dd_line = plot_line_graph(dd_x, dd_y, "#2ca02c", "Temperature (K)", "Second derivative")
dd_minima = plot_scatter_graph(dd_minima_x, dd_minima_y, "#ff7f0e", "Temperature (K)", "Second derivative")
dd_intensities = [estimate_im_from_tm_helper(tm_value, dd_x, dd_y) / 0.7 for tm_value in st.session_state.T_m]
tm_scatter = plot_scatter_graph(st.session_state.T_m, dd_intensities, "#d62728", "Temperature (K)", "Intensity (au)")
st.write(dd_line + dd_minima + tm_scatter)

with st.form("input_form"):
    st.header("Initial values")
    st.session_state.T_m = st.session_state.T_m[:st.session_state.n]

    cols = st.columns(2)
    for i in range(st.session_state.n):
        with cols[0]:
            st.session_state.b[i] = st.number_input(f"b[{i + 1}]", 1.0, 2.0, value=st.session_state.b[i], key=f"b_{i + 1}")
        with cols[1]:
            st.session_state.E[i] = st.number_input(f"E[{i + 1}]", 0.0, value=st.session_state.E[i], key=f"E_{i + 1}")
        # with cols[2]:
        #     st.session_state.I_m[i] = st.number_input(f"I_m[{i + 1}]", 0.0, value=st.session_state.I_m[i], key=f"I_m_{i + 1}")

    st.session_state.scale_factor = st.number_input("Scale factor", value=st.session_state.scale_factor, key="sc_f", help="Divide the intensity values by this factor for more mathematically precise computations.")
    st.session_state.method = st.selectbox("Method for fitting", ["COBYLA", "COBYQA", "SLSQP", "trust-constr"], 2, help="SLSQP is best. COBYQA may not work.")
    st.session_state.prop_coeff_tol = st.number_input("Prop coeff tol", 0.0, 1.0, value=st.session_state.prop_coeff_tol, key="pct", help="E = prop_coeff * T")

    if st.form_submit_button("Submit"):
        st.success("Values updated.")

exec_button = st.button("Execute")
if exec_button:
    init_vals = []
    for i in range(st.session_state.n):
        init_vals.extend([st.session_state.b[i], st.session_state.I_m[i], st.session_state.T_m[i], st.session_state.E[i]])
    init_vals.extend([0.0, 0.0, 0.0])

    fitted_params = execute_helper(
        st.session_state.T,
        st.session_state.intensity,
        st.session_state.n,
        init_vals,
        st.session_state.scale_factor,
        st.session_state.method,
        st.session_state.prop_coeff_tol
    )

    for i in range(st.session_state.n):
        fitted_params[i * 4 + 1] *= st.session_state.scale_factor
    fitted_params[-3] *= st.session_state.scale_factor
    fitted_params[-2] *= st.session_state.scale_factor

    st.header("Fitted parameters")
    for i in range(st.session_state.n):
        b, I_m, T_m, E = fitted_params[i * 4:(i + 1) * 4]
        half_max_left, half_max_right = half_max_temperatures(b, I_m, T_m, E)
        st.write(f"Peak {i + 1}: b = {b:.2f}, I_m = {I_m:.2f}, T_m = {T_m:.2f}, E = {E:.2f}, θ_1 = {half_max_left:.2f}, θ_2 = {half_max_right:.2f}")

    offset = fitted_params[-3]
    exp_coeff = fitted_params[-2]
    exp_power = fitted_params[-1]
    st.write(f"Offset: {offset:.2f}")
    st.write(f"Exp coeff: {exp_coeff:.2f}")
    st.write(f"Exp power: {exp_power:.2f}")

    final_fom = fom_helper(fitted_params, st.session_state.T, st.session_state.intensity)
    st.write(f"<b>Final FOM: {final_fom:.2f}%</b>", unsafe_allow_html=True)

    data = pd.DataFrame({
        "Temperature (K)": st.session_state.T,
        "Intensity (au)": st.session_state.intensity,
        "Fitted Curve": multi_tl_equation_helper(st.session_state.T, fitted_params),
    })

    peak_data = []
    for i in range(st.session_state.n):
        start_idx = i * 4
        end_idx = start_idx + 4
        peak_intensity = tl_equation_helper(st.session_state.T, *fitted_params[start_idx:end_idx])
        peak_data.append(pd.DataFrame({
            "Temperature (K)": st.session_state.T,
            "Intensity (au)": peak_intensity,
            "Peak": f"Peak {i + 1}"
        }))
    peak_data = pd.concat(peak_data)

    fitted_curve = alt.Chart(data).mark_line(color="red", strokeWidth=2).encode(
        x="Temperature (K)",
        y="Fitted Curve",
        tooltip=["Temperature (K)", "Fitted Curve"]
    )
    peaks = alt.Chart(peak_data).mark_line(strokeDash=[4, 2]).encode(
        x="Temperature (K)",
        y="Intensity (au)",
        color=alt.Color("Peak:N", legend=alt.Legend(title="Peaks")),
        tooltip=["Temperature (K)", "Intensity (au)", "Peak"]
    )

    st.write(intensity_line + tm_scatter + fitted_curve + peaks)

    csv_data = "b,I_m,T_m,E,theta_1,theta_2\n"
    for i in range(st.session_state.n):
        b, I_m, T_m, E = fitted_params[i * 4:(i + 1) * 4]
        half_max_left, half_max_right = half_max_temperatures(b, I_m, T_m, E)
        csv_data += f"{b},{I_m},{T_m},{E},{half_max_left},{half_max_right}\n"
    csv_data += f",,,,,,Offset,{offset}\n"
    csv_data += f",,,,,,Exp coeff,{exp_coeff}\n"
    csv_data += f",,,,,,Exp power,{exp_power}\n"
    csv_data += f",,,,,,Final FOM,{final_fom}\n"
    st.download_button("Download optimised parameters", csv_data, "optimised_params.csv", "text/csv", key="download-csv")
