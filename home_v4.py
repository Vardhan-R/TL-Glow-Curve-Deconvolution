from scipy.optimize import minimize
import altair as alt
import csv
import numpy as np
import pandas as pd
import streamlit as st
import warnings

k = 8.617e-5  # Boltzmann constant in eV / K


def constraints(params):
    n = (len(params) - 3) // 4
    b_values = [params[i * 4] for i in range(n)]
    I_m_values = [params[i * 4 + 1] for i in range(n)]
    T_m_values = [params[i * 4 + 2] for i in range(n)]
    E_values = [params[i * 4 + 3] for i in range(n)]
    # E_values_arr = np.array(E_values)
    # sorted_E_values_arr = E_values_arr[np.argsort(T_m)]
    # sorted_E_values_arr[1:]
    # 463 K, 1.2 eV +/- 0.12
    # E - \beta_m T
    prop_coeff_min = (1 - st.session_state.prop_coeff_tol) * 1.2 / 463
    prop_coeff_max = (1 + st.session_state.prop_coeff_tol) * 1.2 / 463
    min_prop_constr = [E - prop_coeff_min * T_m for E, T_m in zip(E_values, T_m_values)]
    max_prop_constr = [prop_coeff_max * T_m - E for E, T_m in zip(E_values, T_m_values)]
    return [b - 1 for b in b_values] + [2 - b for b in b_values] + I_m_values + T_m_values + E_values + params[-3:].tolist() + min_prop_constr + max_prop_constr


def execute(T: np.ndarray, intensity: np.ndarray, n: int, initial_guess: list[float], scale_factor: float, method: str) -> None:
    for i in range(n):
        initial_guess[i * 4 + 1] /= scale_factor
    initial_guess[-3] /= scale_factor
    initial_guess[-2] /= scale_factor

    # Constraints for optimization
    all_constraints = ({"type": "ineq", "fun": constraints})

    # Curve fitting using minimize
    result = minimize(fom, initial_guess, args=(T, intensity / scale_factor), method=method, constraints=all_constraints)

    return result.x


# Define the TL equation
def tl_equation(T, b, I_m, T_m, E):
    Delta = (2 * k * T) / E
    Delta_m = (2 * k * T_m) / E
    Z_m = 1 + (b - 1) * Delta_m
    term1 = I_m * b ** (b / (b - 1)) * np.exp((E / (k * T)) * ((T - T_m) / T_m))
    term2 = ((b - 1) * (1 - Delta)) * ((T / T_m) ** 2) * np.exp((E / (k * T)) * ((T - T_m) / T_m))
    return term1 * (term2 + Z_m) ** (-b / (b - 1))


# Sum of "n" TL equations with an offset
def multi_tl_equation(T, params):
    n = (len(params) - 3) // 4  # Each component has 4 parameters: b, I_m, T_m, E
    result = params[-3] + params[-2] * np.exp(params[-1] * T)
    # result = 0
    # result = params[-3]
    for i in range(n):
        b, I_m, T_m, E = params[i * 4:(i + 1) * 4]
        result += tl_equation(T, b, I_m, T_m, E)
    return result


# FOM calculation
def fom(params, T, intensity):
    model = multi_tl_equation(T, params)
    numerator = np.sum(np.abs(intensity - model))
    denominator = np.sum(intensity)
    return (numerator / denominator) * 100


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    window_size = int(window_size)
    if window_size <= 1:
        return values.copy()
    window = np.ones(window_size) / window_size
    return np.convolve(values, window, mode="same")


def find_double_derivative_zeroes(x: np.ndarray, y: np.ndarray, ma_window_width: int = 1) -> tuple[list, list]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if len(x) < 3:
        return x.tolist(), y.tolist()

    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    smoothed_second_derivative = moving_average(second_derivative, ma_window_width)

    zero_idx = []
    for i in range(len(smoothed_second_derivative) - 1):
        left = smoothed_second_derivative[i]
        right = smoothed_second_derivative[i + 1]
        if left == 0:
            zero_idx.append(i)
        elif left * right < 0:
            zero_idx.append(i)

    peak_x = []
    peak_y = []
    for i in zero_idx:
        left_x, right_x = x[i], x[i + 1]
        left_y, right_y = smoothed_second_derivative[i], smoothed_second_derivative[i + 1]
        if right_y == left_y:
            zero_x = left_x
        else:
            zero_x = left_x - left_y * (right_x - left_x) / (right_y - left_y)
        zero_x = float(np.clip(zero_x, x.min(), x.max()))
        peak_x.append(zero_x)
        peak_y.append(float(np.interp(zero_x, x, y)))

    return peak_x, peak_y


def find_double_derivative_minima(x: np.ndarray, y: np.ndarray, ma_window_width: int = 1) -> tuple[list, list]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return x.tolist(), y.tolist()

    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    smoothed_second_derivative = moving_average(second_derivative, ma_window_width)

    if len(smoothed_second_derivative) < 3:
        min_idx = int(np.argmin(smoothed_second_derivative))
        return [float(x[min_idx])], [float(np.interp(x[min_idx], x, y))]

    minima_idx = []
    for i in range(1, len(smoothed_second_derivative) - 1):
        left = smoothed_second_derivative[i - 1]
        center = smoothed_second_derivative[i]
        right = smoothed_second_derivative[i + 1]
        if center <= left and center <= right:
            minima_idx.append(i)

    if not minima_idx:
        min_idx = int(np.argmin(smoothed_second_derivative))
        minima_idx = [min_idx]

    peak_x = []
    peak_y = []
    for i in minima_idx:
        peak_x.append(float(x[i]))
        peak_y.append(float(np.interp(x[i], x, y)))

    return peak_x, peak_y


def refine_peak_seeds_from_minima(
    x: np.ndarray,
    y: np.ndarray,
    minima_x: list,
    search_width: int,
) -> tuple[list, list]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    search_width = max(1, int(search_width))

    peak_x = []
    peak_y = []
    used_indices: set[int] = set()

    for seed_x in minima_x:
        seed_idx = int(np.argmin(np.abs(x - seed_x)))
        left_idx = max(0, seed_idx - search_width)
        right_idx = min(len(x) - 1, seed_idx + search_width)
        local_idx = left_idx + int(np.argmax(y[left_idx:right_idx + 1]))

        if local_idx in used_indices:
            continue

        used_indices.add(local_idx)
        peak_x.append(float(x[local_idx]))
        peak_y.append(float(y[local_idx]))

    return peak_x, peak_y


def get_smoothed_double_derivative(x: np.ndarray, y: np.ndarray, ma_window_width: int = 1) -> tuple[np.ndarray, np.ndarray, list, list]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return x, np.zeros_like(x), x.tolist(), y.tolist()

    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    smoothed_second_derivative = moving_average(second_derivative, ma_window_width)

    minima_x, minima_y = find_double_derivative_minima(x, y, ma_window_width)
    return x, smoothed_second_derivative, minima_x, minima_y


def plotLineGraph(x: np.ndarray, y: np.ndarray, colour: str, x_label: str = "x", y_label: str = "y") -> alt.Chart:
    source = pd.DataFrame({
    x_label: x,
    y_label: y
    })

    # c = alt.Chart(source).mark_line(point=True).encode(
    #     x=alt.X("x:O", title=x_label),
    #     y=alt.Y("y:O", title=y_label),
    #     color=alt.Color("symbol:N")
    # )

    c = alt.Chart(source).mark_line(color=colour).encode(
        x=x_label,
        y=y_label,
    ).interactive()

    return c


def plotScatterGraph(x: np.ndarray, y: np.ndarray, colour: str, x_label: str = "x", y_label: str = "y") -> alt.Chart:
    source = pd.DataFrame({
    x_label: x,
    y_label: y
    })

    # c = alt.Chart(source).mark_line(point = True).encode(
    #     x=alt.X("x:O", title=x_label),
    #     y=alt.Y("y:O", title=y_label),
    #     color=alt.Color("symbol:N")
    # )

    c = alt.Chart(source).mark_circle(color=colour, size=60).encode(
        x=x_label,
        y=y_label,
    ).interactive()

    return c


# Update session state when n is changed
def update_n() -> None:
    n = int(st.session_state.n_input)
    diff = n - len(st.session_state.b)
    if diff > 0:  # Add new rows
        st.session_state.b.extend([1.6] * diff)
        st.session_state.I_m.extend([50000.0] * diff)
        st.session_state.T_m.extend([500.0] * diff)
        st.session_state.E.extend([np.random.normal(1, 0.1)] * diff)
    elif diff < 0:  # Remove excess rows
        st.session_state.b = st.session_state.b[:n]
        st.session_state.I_m = st.session_state.I_m[:n]
        st.session_state.T_m = st.session_state.T_m[:n]
        st.session_state.E = st.session_state.E[:n]
    st.session_state.n = n


def initialise_from_data() -> None:
    minima_T, _ = find_double_derivative_minima(
        st.session_state.T,
        st.session_state.intensity,
        st.session_state.dd_ma_window_width,
    )
    st.session_state.peak_T, st.session_state.peak_I = refine_peak_seeds_from_minima(
        st.session_state.T,
        st.session_state.intensity,
        minima_T,
        st.session_state.dd_ma_window_width,
    )
    st.session_state.n = len(st.session_state.peak_T)
    st.session_state.b = [1.6] * st.session_state.n
    st.session_state.I_m = st.session_state.peak_I.copy()
    st.session_state.T_m = st.session_state.peak_T.copy()
    st.session_state.E = np.random.normal(1, 0.1, st.session_state.n).tolist()
    st.session_state.scale_factor = 500.0
    st.session_state.method = "SLSQP"
    st.session_state.prop_coeff_tol = 0.1
    st.session_state.located_maxima = True


warnings.filterwarnings("ignore")

st.set_page_config("TL Glow Curve Deconvolution")

st.title("TL Glow Curve Deconvolution")

# Initialise session state variables
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False
    st.session_state.located_maxima = False
    st.session_state.n = 1  # Default number of peaks
    st.session_state.b = [0.0]
    st.session_state.I_m = [0.0]
    st.session_state.T_m = [0.0]
    st.session_state.E = [0.0]
    st.session_state.method = "SLSQP"
    st.session_state.prop_coeff_tol = 0.1
    st.session_state.dd_ma_window_width = 25

if not st.session_state.uploaded:
    cols = st.columns([3, 1])
    with cols[0]:
        st.write("Upload the file containing the data (in csv format only). There file must not have any headers. The first column must contain the temperature values (in K), and the second column must contain the intensity values (in a.u.).")
        with st.form("upload"):
            file = st.file_uploader("Upload data file", "csv", False)
            submitted = st.form_submit_button("Upload", icon='🚀')
            if file and submitted:
                with open("temp_file.csv", 'wb') as fp:
                    fp.write(file.getvalue())
                file.close()
                with open("temp_file.csv", 'r') as fp:
                    reader = csv.reader(fp)
                    data = np.array([list(map(np.float64, row)) for row in reader])

                st.session_state.T, st.session_state.intensity = data[:, 0], data[:, 1]

                st.session_state.uploaded = True
                st.rerun()
    with cols[1]:
        st.image("images/csv_file_example.png", "Example of the data file", width="stretch")

    # with open("data.csv", 'r') as fp:
    #     reader = csv.reader(fp)
    #     data = np.array([list(map(np.float64, row)) for row in reader])

    # st.session_state.T, st.session_state.intensity = data[:, 0], data[:, 1]

    # st.session_state.uploaded = True
    # st.rerun()
else:
    if not st.session_state.located_maxima:
        initialise_from_data()
        st.rerun()
    else:
        # Display the input boxes in four columns

        # Input for "n"
        st.number_input("n (number of peaks)", 1, value=st.session_state.n, key="n_input", on_change=update_n)

        with st.form("input_form"):
            st.header("Initial values")
            cols = st.columns(4)
            for i in range(st.session_state.n):
                with cols[0]:
                    st.session_state.b[i] = st.number_input(f"b[{i + 1}]", 1.0, 2.0, value=st.session_state.b[i], key=f"b_{i + 1}")
                with cols[1]:
                    st.session_state.I_m[i] = st.number_input(f"I_m[{i + 1}]", 0.0, value=st.session_state.I_m[i], key=f"I_m_{i + 1}")
                with cols[2]:
                    st.session_state.T_m[i] = st.number_input(f"T_m[{i + 1}]", 0.0, value=st.session_state.T_m[i], key=f"T_m_{i + 1}")
                with cols[3]:
                    st.session_state.E[i] = st.number_input(f"E[{i + 1}]", 0.0, value=st.session_state.E[i], key=f"E_{i + 1}")

            st.session_state.scale_factor = st.number_input("Scale factor", value=st.session_state.scale_factor, key="sc_f", help="Divide the intensity values by this factor for more mathematically precise computations.")

            st.session_state.method = st.selectbox("Method for fitting", ["COBYLA", "COBYQA", "SLSQP", "trust-constr"], 2, help="SLSQP is best. COBYQA may not work.")

            st.session_state.prop_coeff_tol = st.number_input("Prop coeff tol", 0.0, 1.0, value=st.session_state.prop_coeff_tol, key="pct", help="E = prop_coeff * T")
            st.session_state.dd_ma_window_width = st.number_input(
                "Double derivative MA window width",
                min_value=1,
                value=int(st.session_state.dd_ma_window_width),
                step=1,
                help="Moving average window width applied to the double derivative before locating zero crossings.",
            )

            if st.form_submit_button("Recompute initial values"):
                initialise_from_data()
                st.success("Initial values recomputed from the double derivative.")
                st.rerun()

            # Submit button
            submitted = st.form_submit_button("Submit")
            if submitted:
                st.success("Values updated")

        c1 = plotLineGraph(st.session_state.T, st.session_state.intensity, "#17becf", "Temperature (K)", "Intensity (au)")
        # c2 = plotScatterGraph(st.session_state.peak_T, st.session_state.peak_I, "#d62728", "Temperature (K)", "Intensity (au)")
        c2 = plotScatterGraph(st.session_state.T_m, st.session_state.I_m, "#d62728", "Temperature (K)", "Intensity (au)")
        st.write(c1 + c2)

        dd_x, dd_y, dd_minima_x, dd_minima_y = get_smoothed_double_derivative(
            st.session_state.T,
            st.session_state.intensity,
            st.session_state.dd_ma_window_width,
        )
        c3 = plotLineGraph(dd_x, dd_y, "#2ca02c", "Temperature (K)", "Second derivative")
        # c4 = plotScatterGraph(dd_minima_x, dd_minima_y, "#ff7f0e", "Temperature (K)", "Second derivative")
        # st.write(c3 + c4)
        st.write(c3)

        exec_button = st.button("Execute")
        if exec_button:
            init_vals = []
            for i in range(st.session_state.n):
                init_vals.extend([st.session_state.b[i], st.session_state.I_m[i], st.session_state.T_m[i], st.session_state.E[i]])
            init_vals.extend([0.0, 0.0, 0.0])  # For the base term

            fitted_params = execute(st.session_state.T, st.session_state.intensity, st.session_state.n, init_vals, st.session_state.scale_factor, st.session_state.method)

            for i in range(st.session_state.n):
                fitted_params[i * 4 + 1] *= st.session_state.scale_factor
            fitted_params[-3] *= st.session_state.scale_factor
            fitted_params[-2] *= st.session_state.scale_factor

            st.header("Fitted parameters")
            for i in range(st.session_state.n):
                b, I_m, T_m, E = fitted_params[i * 4:(i + 1) * 4]
                st.write(f"Peak {i + 1}: b = {b:.2f}, I_m = {I_m:.2f}, T_m = {T_m:.2f}, E = {E:.2f}")

            offset = fitted_params[-3]
            exp_coeff = fitted_params[-2]
            exp_power = fitted_params[-1]
            st.write(f"Offset: {offset:.2f}")
            st.write(f"Exp coeff: {exp_coeff:.2f}")
            st.write(f"Exp power: {exp_power:.2f}")

            # Display the final FOM value
            final_fom = fom(fitted_params, st.session_state.T, st.session_state.intensity)
            st.write(f"<b>Final FOM: {final_fom:.2f}%</b>", unsafe_allow_html=True)

            # Data preparation
            data = pd.DataFrame({
                "Temperature (K)": st.session_state.T,
                "Intensity (au)": st.session_state.intensity,
                "Fitted Curve": multi_tl_equation(st.session_state.T, fitted_params)
            })

            # Create individual peak data
            peak_data = []
            for i in range(st.session_state.n):
                start_idx = i * 4
                end_idx = start_idx + 4
                peak_intensity = tl_equation(st.session_state.T, *fitted_params[start_idx:end_idx])
                peak_data.append(pd.DataFrame({
                    "Temperature (K)": st.session_state.T,
                    "Intensity (au)": peak_intensity,
                    "Peak": f"Peak {i + 1}"
                }))

            peak_data = pd.concat(peak_data)

            # # Scatter plot for data
            # scatter = alt.Chart(data).mark_circle(size=30, color="blue").encode(
            #     x="Temperature",
            #     y="Intensity",
            #     tooltip=["Temperature", "Intensity"]
            # ).properties(title="Fitting TL Glow Curves with FOM")

            # Line plot for fitted curve
            fitted_curve = alt.Chart(data).mark_line(color="red", strokeWidth=2).encode(
                x="Temperature (K)",
                y="Fitted Curve",
                tooltip=["Temperature (K)", "Fitted Curve"]
            )

            # Line plots for individual peaks
            peaks = alt.Chart(peak_data).mark_line(strokeDash=[4, 2]).encode(
                x="Temperature (K)",
                y="Intensity (au)",
                color=alt.Color("Peak:N", legend=alt.Legend(title="Peaks")),
                tooltip=["Temperature (K)", "Intensity (au)", "Peak"]
            )

            # Combine plots
            chart = c1 + fitted_curve + peaks

            st.write(chart)

            # Download the optimised params as CSV
            peaks = alt.Chart(peak_data).mark_line(strokeDash=[4, 2]).encode(
                x="Temperature (K)",
                y="Intensity (au)",
                color=alt.Color("Peak:N", legend=alt.Legend(title="Peaks")),
                tooltip=["Temperature (K)", "Intensity (au)", "Peak"]
            )

            # Combine plots
            chart = c1 + fitted_curve + peaks

            st.write(chart)

            # Download the optimised params as CSV
            csv_data = "b,I_m,T_m,E\n"
            for i in range(st.session_state.n):
                b, I_m, T_m, E = fitted_params[i * 4:(i + 1) * 4]
                csv_data += f"{b},{I_m},{T_m},{E}\n"

            # Add the , Offset, Exp coeff, Exp power, Final FOM
            csv_data += f",,,Offset,{offset}\n"
            csv_data += f",,,Exp coeff,{exp_coeff}\n"
            csv_data += f",,,Exp power,{exp_power}\n"
            csv_data += f",,,Final FOM,{final_fom}\n"

            st.download_button("Download Optimised Parameters", csv_data, "optimised_params.csv", "text/csv", key="download-csv")
