from scipy.optimize import minimize
from scipy.optimize import root_scalar
import numpy as np
import streamlit as st

k = 8.617e-5  # Boltzmann constant in eV / K


def constraints(params: list[float], prop_coeff_tol: float) -> list[float]:
    n = (len(params) - 3) // 4
    b_values = [params[i * 4] for i in range(n)]
    I_m_values = [params[i * 4 + 1] for i in range(n)]
    T_m_values = [params[i * 4 + 2] for i in range(n)]
    E_values = [params[i * 4 + 3] for i in range(n)]
    prop_coeff_min = (1 - prop_coeff_tol) * 1.2 / 463
    prop_coeff_max = (1 + prop_coeff_tol) * 1.2 / 463
    min_prop_constr = [E - prop_coeff_min * T_m for E, T_m in zip(E_values, T_m_values)]
    max_prop_constr = [prop_coeff_max * T_m - E for E, T_m in zip(E_values, T_m_values)]
    return [b - 1 for b in b_values] + [2 - b for b in b_values] + I_m_values + T_m_values + E_values + params[-3:].tolist() + min_prop_constr + max_prop_constr


def execute(T: np.ndarray, intensity: np.ndarray, n: int, initial_guess: list[float], scale_factor: float, method: str, prop_coeff_tol: float) -> np.ndarray:
    for i in range(n):
        initial_guess[i * 4 + 1] /= scale_factor
    initial_guess[-3] /= scale_factor
    initial_guess[-2] /= scale_factor
    all_constraints = ({"type": "ineq", "fun": lambda params: constraints(params, prop_coeff_tol)})
    result = minimize(fom, initial_guess, args=(T, intensity / scale_factor), method=method, constraints=all_constraints)
    return result.x


def tl_equation(T, b, I_m, T_m, E):
    Delta = (2 * k * T) / E
    Delta_m = (2 * k * T_m) / E
    Z_m = 1 + (b - 1) * Delta_m
    term1 = I_m * b ** (b / (b - 1)) * np.exp((E / (k * T)) * ((T - T_m) / T_m))
    term2 = ((b - 1) * (1 - Delta)) * ((T / T_m) ** 2) * np.exp((E / (k * T)) * ((T - T_m) / T_m))
    return term1 * (term2 + Z_m) ** (-b / (b - 1))


def multi_tl_equation(T, params):
    n = (len(params) - 3) // 4
    result = params[-3] + params[-2] * np.exp(params[-1] * T)
    for i in range(n):
        b, I_m, T_m, E = params[i * 4:(i + 1) * 4]
        result += tl_equation(T, b, I_m, T_m, E)
    return result


def fom(params, T, intensity):
    model = multi_tl_equation(T, params)
    numerator = np.sum(np.abs(intensity - model))
    denominator = np.sum(intensity)
    return (numerator / denominator) * 100


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    window_size = max(1, int(window_size))
    if window_size <= 1:
        return values.copy()
    return np.convolve(values, np.ones(window_size) / window_size, mode="same")


def find_double_derivative_minima(x: np.ndarray, y: np.ndarray, ma_window_width: int = 1) -> tuple[list, list]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x.tolist(), y.tolist()

    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    smoothed_second_derivative = moving_average(second_derivative, ma_window_width)

    minima_idx = []
    for i in range(1, len(smoothed_second_derivative) - 1):
        left = smoothed_second_derivative[i - 1]
        center = smoothed_second_derivative[i]
        right = smoothed_second_derivative[i + 1]
        if center <= left and center <= right:
            minima_idx.append(i)

    if not minima_idx:
        minima_idx = [int(np.argmin(smoothed_second_derivative))]

    peak_x = [float(x[i]) for i in minima_idx]
    peak_y = [float(np.interp(x[i], x, smoothed_second_derivative)) for i in minima_idx]
    return peak_x, peak_y


def get_double_derivative_curve(x: np.ndarray, y: np.ndarray, ma_window_width: int = 1) -> tuple[np.ndarray, np.ndarray, list, list]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x, np.zeros_like(x), x.tolist(), y.tolist()

    first_derivative = np.gradient(y, x)
    second_derivative = np.gradient(first_derivative, x)
    smoothed_second_derivative = moving_average(second_derivative, ma_window_width)
    minima_x, minima_y = find_double_derivative_minima(x, y, ma_window_width)
    return x, smoothed_second_derivative, minima_x, minima_y


def estimate_im_from_tm(tm_value: float, temperatures: np.ndarray, intensity: np.ndarray) -> float:
    temperatures = np.asarray(temperatures, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    tm_value = float(tm_value)

    if len(temperatures) == 0:
        return 0.0
    if len(temperatures) == 1:
        return float(0.7 * intensity[0])

    order = np.argsort(temperatures)
    sorted_temperatures = temperatures[order]
    sorted_intensity = intensity[order]

    insert_idx = int(np.searchsorted(sorted_temperatures, tm_value))
    if insert_idx <= 0:
        left_idx, right_idx = 0, 1
    elif insert_idx >= len(sorted_temperatures):
        left_idx, right_idx = len(sorted_temperatures) - 2, len(sorted_temperatures) - 1
    else:
        left_idx, right_idx = insert_idx - 1, insert_idx

    left_t = sorted_temperatures[left_idx]
    right_t = sorted_temperatures[right_idx]
    left_i = sorted_intensity[left_idx]
    right_i = sorted_intensity[right_idx]

    if right_t == left_t:
        interpolated_intensity = float(left_i)
    else:
        interpolated_intensity = float(left_i + (tm_value - left_t) * (right_i - left_i) / (right_t - left_t))

    return 0.7 * interpolated_intensity


def half_max_temperatures(b, I_m, T_m, E):
    f = lambda T: tl_equation(T, b, I_m, T_m, E) - I_m / 2

    left = root_scalar(
        f,
        bracket=[max(1e-6, T_m - 500), T_m],
        method="brentq",
    ).root

    right = root_scalar(
        f,
        bracket=[T_m, T_m + 500],
        method="brentq",
    ).root

    return left, right
