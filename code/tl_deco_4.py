import numpy as np
import matplotlib.pyplot as plt
import csv

def gaussian_kernel(window_size, sigma=None):
    """
    Generates a 1D Gaussian kernel of width `window_size`.
    If `sigma` is not provided, it is estimated as (window_size / 6).
    """
    # Estimate sigma if not provided
    if sigma is None:
        sigma = window_size / 6.0  # Typically, window_size / 6 is a good estimate

    # Create an array of positions
    x = np.linspace(-window_size // 2, window_size // 2, window_size)

    # Compute the Gaussian function
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    
    # Normalize the kernel so that the sum of all values is 1
    kernel = kernel / np.sum(kernel)

    return kernel

# Read CSV data and convert to numpy arrays
with open("C:\\Users\\vrdhn\\Desktop\\data.csv", 'r') as fp:
    reader = csv.reader(fp)
    data = np.array([list(map(int, row)) for row in reader], np.float64)

# Scaling factor and extracting T and intensity
scale_factor = 400
T, intensity = data[:, 0], data[:, 1] / scale_factor
window_size = 5  # Adjust the window size as needed
intensity = np.convolve(intensity, np.ones(window_size) / window_size, 'same')

# Compute derivatives
dT = np.diff(T)
dIntensity = np.diff(intensity)

# First derivative
first_derivative = dIntensity / dT

# Moving average of the first derivative
window_size = 5  # Adjust the window size as needed
moving_avg_first_derivative = np.convolve(first_derivative, np.ones(window_size) / window_size, mode='same')

# Adjust T for plotting the moving average of the first derivative
T_avg = T[1:]  # Align T with the moving average data

# Second derivative (using the smoothed first derivative)
dFirstDerivative = np.diff(moving_avg_first_derivative)
dT_first_derivative = (dT[:-1] + dT[1:]) / 2  # Average T intervals for second derivative
second_derivative = dFirstDerivative / dT_first_derivative

idx = [i + 1 for i, (T_0, T_1) in enumerate(zip(moving_avg_first_derivative[:-1], moving_avg_first_derivative[1:])) if T_0 > 0 and T_1 < 0]
peak_x = [T[i] for i in idx]
peak_y = [intensity[i] for i in idx]

# Plotting the original data and derivatives
plt.figure(figsize=(10, 6))

# Plot the original data
plt.subplot(3, 1, 1)
plt.plot(T, intensity, label='Original Data')
plt.scatter(peak_x, peak_y)
plt.title('Original Data')
plt.xlabel('T')
plt.ylabel('Intensity')
plt.grid(True)

# Plot the first derivative (moving average)
plt.subplot(3, 1, 2)
plt.plot(T_avg, moving_avg_first_derivative, label='Moving Average of First Derivative', color='orange')
plt.title('First Derivative (Moving Average)')
plt.xlabel('T')
plt.ylabel('d(Intensity)/dT (Smoothed)')
plt.grid(True)

# Plot the second derivative
plt.subplot(3, 1, 3)
plt.plot(T[2:], second_derivative, label='Second Derivative', color='green')
plt.title('Second Derivative')
plt.xlabel('T')
plt.ylabel('d²(Intensity)/dT²')
plt.grid(True)

plt.tight_layout()
plt.show()
