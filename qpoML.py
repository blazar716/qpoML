import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from astropy.stats import LombScargle as PDM

# Generate synthetic light curve data with a 10-day sinusoidal period and Gaussian noise
np.random.seed(42)
time = np.linspace(0, 100, 1000)
true_period = 10
flux = np.sin(2 * np.pi * time / true_period) + 0.1 * np.random.normal(size=len(time))

# Add some noise to the data
flux += 0.1 * np.random.normal(size=len(time))

# Use Lomb-Scargle to get an initial periodogram and identify the peak
frequency, power = LombScargle(time, flux).autopower()
best_frequency = frequency[np.argmax(power)]
initial_best_period = 1 / best_frequency

# Print the initial best period
print(f'Initial best period from Lomb-Scargle: {initial_best_period:.2f} days')

# Plot the periodogram
plt.figure(figsize=(10, 4))
plt.plot(1 / frequency, power)
plt.axvline(initial_best_period, color='r', linestyle='--', label=f'Initial Best Period: {initial_best_period:.2f} days')
plt.xlabel('Period (days)')
plt.ylabel('Power')
plt.legend()
plt.title('Lomb-Scargle Periodogram')
plt.show()

# Use PDM to refine the period search
min_period = 5
max_period = 15
periods = np.linspace(min_period, max_period, 1000)
pdm_scores = np.zeros_like(periods)

for i, period in enumerate(periods):
    phase = (time / period) % 1
    bins = np.linspace(0, 1, 10)
    digitized = np.digitize(phase, bins)
    bin_means = [flux[digitized == j].mean() for j in range(1, len(bins))]
    bin_vars = [flux[digitized == j].var() for j in range(1, len(bins))]
    pdm_scores[i] = np.sum(bin_vars) / np.var(flux)

best_period_pdm = periods[np.argmin(pdm_scores)]

# Print the best period from PDM
print(f'Best period from PDM: {best_period_pdm:.2f} days')

# Plot PDM results
plt.figure(figsize=(10, 4))
plt.plot(periods, pdm_scores)
plt.axvline(best_period_pdm, color='r', linestyle='--', label=f'Best Period: {best_period_pdm:.2f} days')
plt.xlabel('Period (days)')
plt.ylabel('PDM Score')
plt.legend()
plt.title('PDM Period Search')
plt.show()

# Plot the folded light curve using the best period from PDM
phase = (time / best_period_pdm) % 1
plt.figure(figsize=(10, 4))
plt.scatter(phase, flux, s=10)
plt.xlabel('Phase')
plt.ylabel('Flux')
plt.title('Folded Light Curve using Best Period from PDM')
plt.show()

