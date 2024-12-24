import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv

# Constants
k = 1  # Boltzmann constant in eV/K

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
    n = (len(params) - 1) // 4  # Each component has 4 parameters: b, I_m, T_m, E
    offset = params[-1]  # Last parameter is the offset
    result = offset
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

# Read CSV data and convert to numpy arrays
with open("C:\\Users\\vrdhn\\Desktop\\data.csv", 'r') as fp:
    reader = csv.reader(fp)
    data = np.array([list(map(int, row)) for row in reader], np.float64)

T, intensity = data[:, 0], data[:, 1]

# Normalize data
T_min, T_max = 300, 800
intensity_min, intensity_max = 5000, 50000
T_normalized = (T - T_min) / (T_max - T_min)
intensity_shifted = intensity - intensity_min
intensity_normalized = intensity_shifted / (intensity_max - intensity_min)

# Initial guesses for fitting
n = 3  # Number of peaks
initial_guess = [1.9, 0, 1, 1.0] * n + [0.0]  # Add initial guess for offset

# Curve fitting using minimize
result = minimize(fom, initial_guess, args=(T_normalized, intensity_normalized), method="Nelder-Mead")

# Extract fitted parameters
fitted_params = result.x

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(T_normalized, intensity_normalized, s=10, label="Data", color="blue")
plt.plot(T_normalized, multi_tl_equation(T_normalized, fitted_params), label="Fitted Curve", color="red")
for i in range(n):
    plt.plot(T_normalized, tl_equation(T_normalized, *fitted_params[i * 4:(i + 1) * 4]), linestyle="--", label=f"Peak {i + 1}")
plt.xlabel("Normalized Temperature")
plt.ylabel("Normalized Intensity")
plt.legend()
plt.title("Fitting Normalized TL Glow Curves with FOM")
plt.show()

# Print fitted parameters
print("Fitted parameters:")
for i in range(n):
    b, I_m, T_m, E = fitted_params[i * 4:(i + 1) * 4]
    print(f"Peak {i + 1}: b = {b:.2f}, I_m = {I_m:.2f}, T_m = {T_m:.2f}, E = {E:.2f}")

offset = fitted_params[-1]
print(f"Offset: {offset:.2f}")

# Print final FOM value
final_fom = fom(fitted_params, T_normalized, intensity_normalized)
print(f"Final FOM: {final_fom:.2f}%")
