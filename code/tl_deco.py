from scipy.optimize import curve_fit
import csv
import csv
import matplotlib.pyplot as plt
import numpy as np

# Read CSV data and convert to numpy arrays
with open("C:\\Users\\vrdhn\\Desktop\\data.csv", 'r') as fp:
    reader = csv.reader(fp)
    data = np.array([list(map(int, row)) for row in reader], np.float64)

temperature, intensity = data[:, 0], data[:, 1]

# Define a Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Function to iteratively fit and subtract Gaussians
def iterativeGaussianFit(x, y, n):
    residual = y.copy()
    fitted_params_list = []

    for i in range(n):
        # Initial guess for parameters: amplitude=1, mean=middle of x, stddev=range of x / 10
        initial_guess = [max(residual), np.mean(x), (x[-1] - x[0]) / 10]

        # Fit the data with a single Gaussian
        popt, _ = curve_fit(gaussian, x, residual, p0=initial_guess)
        fitted_params_list.append(popt)

        # Subtract the fitted Gaussian from the residual
        residual -= gaussian(x, *popt)

        # Plot the intermediate results
        plt.plot(x, gaussian(x, *popt), linestyle="--", label=f"Gaussian {i+1}")

    return fitted_params_list, residual

# Define the sum of n Gaussian functions
def multiGaussian(x, *params):
    n = len(params) // 3  # Each Gaussian has 3 parameters: amplitude, mean, stddev
    result = np.zeros_like(x, dtype=np.float64)  # Ensure result is float64
    for i in range(n):
        amplitude = params[i * 3]
        mean = params[i * 3 + 1]
        stddev = params[i * 3 + 2]
        result += gaussian(x, amplitude, mean, stddev)
    return result

n = 3
fitted_params_list, residual = iterativeGaussianFit(temperature, intensity, n)

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(temperature, intensity, s=10, label="Data", color="blue")
plt.plot(temperature, sum([gaussian(temperature, *params) for params in fitted_params_list]), label="Fitted Curve", color="red")
plt.xlabel("Temperature")
plt.ylabel("Intensity")
plt.legend()
plt.title("Iterative Gaussian Fitting")
plt.show()

# Print fitted parameters
print("Fitted parameters:")
for i, params in enumerate(fitted_params_list):
    print(f"Gaussian {i+1}: Amplitude = {params[0]:.2f}, Mean = {params[1]:.2f}, Stddev = {params[2]:.2f}")

# n = 5
# initial_guess = []
# for i in range(n):
#     initial_guess.extend([1, (i + 1) * (temperature[-1] / (n + 1)), 10])

# # Fit the data
# popt, pcov = curve_fit(lambda x, *params: multi_gaussian(x, *params), temperature, intensity, p0=initial_guess)

# # Extract fitted parameters
# fitted_params = popt


# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.scatter(temperature, intensity, s=10, label="Data", color="blue")
# plt.plot(temperature, multi_gaussian(temperature, *fitted_params), label="Fitted Curve", color="red")

# # Plot individual Gaussians
# for i in range(n):
#     amplitude = fitted_params[i * 3]
#     mean = fitted_params[i * 3 + 1]
#     stddev = fitted_params[i * 3 + 2]
#     plt.plot(temperature, gaussian(temperature, amplitude, mean, stddev), linestyle="--", label=f"Gaussian {i+1}")

# plt.xlabel("Temperature")
# plt.ylabel("Intensity")
# plt.legend()
# plt.title("Fitting Temperature vs. Intensity with Gaussian Curves")
# plt.show()

# # Print fitted parameters
# print("Fitted parameters:")
# for i in range(n):
#     print(f"Gaussian {i+1}: Amplitude = {fitted_params[i * 3]:.2f}, Mean = {fitted_params[i * 3 + 1]:.2f}, Stddev = {fitted_params[i * 3 + 2]:.2f}")
