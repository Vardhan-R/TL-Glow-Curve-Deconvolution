from scipy.optimize import minimize
import csv
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Constants
k = 8.617e-5  # Boltzmann constant in eV/K

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

# Constraint function for "b" values to lie between 1 and 2 and base values >= 0
def constraints(params):
    n = (len(params) - 3) // 4
    b_values = [params[i * 4] for i in range(n)]
    return [b - 1 for b in b_values] + [2 - b for b in b_values] + params[-3:].tolist()

def find_maxima(x, y, window_size=5):
    y = np.convolve(y, np.ones(window_size) / window_size, "same")

    # Compute derivatives
    dx = np.diff(x)
    dy = np.diff(y)

    # First derivative
    first_derivative = dy / dx

    # Moving average of the first derivative
    moving_avg_first_derivative = np.convolve(first_derivative, np.ones(window_size) / window_size, mode="same")

    # # Adjust T for plotting the moving average of the first derivative
    # T_avg = T[1:]  # Align T with the moving average data

    # # Second derivative (using the smoothed first derivative)
    # dFirstDerivative = np.diff(moving_avg_first_derivative)
    # dx_first_derivative = (dx[:-1] + dx[1:]) / 2  # Average T intervals for second derivative
    # second_derivative = dFirstDerivative / dx_first_derivative

    idx = [i + 1 for i, (x_0, x_1) in enumerate(zip(moving_avg_first_derivative[:-1], moving_avg_first_derivative[1:])) if x_0 > 0 and x_1 < 0]
    peak_x = [x[i] for i in idx]
    peak_y = [y[i] for i in idx]

    return peak_x, peak_y

# Read CSV data and convert to numpy arrays
with open("C:\\Users\\vrdhn\\Desktop\\data.csv", 'r') as fp:
    reader = csv.reader(fp)
    data = np.array([list(map(int, row)) for row in reader], np.float64)

scale_factor = 500
T, intensity = data[:, 0], data[:, 1] / scale_factor

peak_T, peak_I = find_maxima(T, intensity)

# Initial guesses for fitting
n = len(peak_T) # Number of peaks
# b, I_m, T_m, E
initial_guess = [1.6, 85000 / scale_factor, 357, 1.0,   # 170
                 1.6, 85000 / scale_factor, 393, 1.0,   # 170
                 1.6, 50000 / scale_factor, 427, 1.0,   # 100
                 1.6, 50000 / scale_factor, 517, 1.0,   # 100
                 1.6, 12000 / scale_factor, 671, 1.0] + [1 / scale_factor, 1 / scale_factor, 0.001]

initial_guess = [None] * (4 * n + 3)

for i in range(4 * n):
    match i % 4:
        case 0: # b
            initial_guess[i] = 1.6
        case 1: # I_m
            initial_guess[i] = peak_I[i // 4]
        case 2: # T_m
            initial_guess[i] = peak_T[i // 4]
        case 3: # E
            initial_guess[i] = np.random.normal(1, 0.05)

initial_guess[-3] = 0.1 / scale_factor
initial_guess[-2] = 0.1 / scale_factor
initial_guess[-1] = 0.001

initial_guess = initial_guess[:-3] + [1.6, 50000 / scale_factor, 427, 1.0] + initial_guess[-3:]
n += 1

method = ["COBYLA", "COBYQA", "SLSQP", "trust-constr"][2]

# Print fitted parameters
print("Initial parameters:")
for i in range(n):
    b, I_m, T_m, E = initial_guess[i * 4:(i + 1) * 4]
    print(f"Peak {i + 1}: b = {b:.2f}, I_m = {I_m:.2f}, T_m = {T_m:.2f}, E = {E:.2f}")

print(f"Offset: {initial_guess[-3]:.2f}")
print(f"Exp coeff: {initial_guess[-2]:.2f}")
print(f"Exp power: {initial_guess[-1]:.2f}")
print(f"Scale factor: {scale_factor}")
print(f"Method: {method}")

# Constraints for optimization
all_constraints = ({"type": "ineq", "fun": constraints})

# Curve fitting using minimize
result = minimize(fom, initial_guess, args=(T, intensity), method=method, constraints=all_constraints)

# Extract fitted parameters
fitted_params = result.x

intensity *= scale_factor
for i in range(n):
    fitted_params[i * 4 + 1] *= scale_factor
fitted_params[-3] *= scale_factor
fitted_params[-2] *= scale_factor

# Print fitted parameters
print("Fitted parameters:")
for i in range(n):
    b, I_m, T_m, E = fitted_params[i * 4:(i + 1) * 4]
    print(f"Peak {i + 1}: b = {b:.2f}, I_m = {I_m:.2f}, T_m = {T_m:.2f}, E = {E:.2f}")

print(f"Offset: {fitted_params[-3]:.2f}")
print(f"Exp coeff: {fitted_params[-2]:.2f}")
print(f"Exp power: {fitted_params[-1]:.2f}")

# Print final FOM value
final_fom = fom(fitted_params, T, intensity)
print(f"Final FOM: {final_fom:.2f}%")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(T, intensity, s=10, label="Data", color="blue")
plt.plot(T, multi_tl_equation(T, fitted_params), label="Fitted Curve", color="red")
for i in range(n):
    plt.plot(T, tl_equation(T, *fitted_params[i * 4:(i + 1) * 4]), linestyle="--", label=f"Peak {i + 1}")
plt.xlabel("Temperature (K)")
plt.ylabel("Intensity (a.u.)")
plt.legend()
plt.title("Fitting TL Glow Curves with FOM")
plt.show()
