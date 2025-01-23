import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Error values
errors = np.array([0.5, -0.7, 1.2, -1.5, 0.3])

# Plotting the histogram
plt.figure(figsize=(8, 6))
plt.hist(errors, bins=5, color='skyblue', edgecolor='black', alpha=0.7, density=True)

# Adding the normal distribution curve
x_vals = np.linspace(errors.min(), errors.max(), 100)
pdf_vals = norm.pdf(x_vals, loc=np.mean(errors), scale=np.std(errors))
plt.plot(x_vals, pdf_vals, color='red', lw=2, label='Normal Distribution Curve')

# Adding labels and title
plt.title("Histogram of Errors", fontsize=14)
plt.xlabel("Error Values", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
