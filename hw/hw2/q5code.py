import matplotlib.pyplot as plt
import numpy as np

## Given:
N_SAMPLES = 10000
pi = lambda x: np.exp(-x) if x >= 0 else 0.0
x = 3
# Define:
rng = np.random.default_rng(303)
normal = lambda x: rng.normal(x)
Q = lambda x, y: np.exp(-((y - x) ** 2) / 2) / np.sqrt(2 * np.pi)
alpha = lambda x, y: min(1, pi(y) * Q(y, x) / pi(x) / Q(x, y))
# Although the Qs above cancel out, I included them for demonstration.
## Iteration:
samples = [x]
while len(samples) < N_SAMPLES:
    y = normal(x)
    if y == x or np.random.rand() < alpha(x, y):
        x = y  # Walk the Markov chain based on the sampling rate.
    if x >= 0:  # Discard negative samples.
        samples.append(x)
# Plot histogram:
hists, bins, _ = plt.hist(samples, density=True)
[plt.text(b, h, f"{h:.2f}") for b, h in zip(bins, hists)]
plt.title("Histogram of Samples from Metropolis-Hasting Method")
plt.xlabel("x")
plt.ylabel("Density")
plt.grid(True)
plt.show()
