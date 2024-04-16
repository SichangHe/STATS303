import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig, norm

X = np.array([[-1, 0], [0, 0], [2, 1], [0, 1], [-1, -2]])
X_mean = np.mean(X, axis=0)
X_tilde = X - X_mean
cov_X = np.cov(X_tilde.T)
S = cov_X / norm(cov_X)

lambdas, u_vecs = eig(S)
sorted_indices = np.argsort(lambdas)[::-1]
u_vecs = u_vecs[:, sorted_indices]
u_1 = u_vecs[:, 0]

X_projected = X_tilde @ u_vecs
X_hat = X_projected[:, 0]
print(f"Transformed data to 1 dimension:\n{X_hat}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# X.
axes[0].scatter(X[:, 0], X[:, 1], marker="x")
axes[0].set_title("Original Data")
axes[0].set_xlabel("x1")
axes[0].set_ylabel("x2")

# Transformed.
axes[1].scatter(
    X_projected[:, 0],
    X_projected[:, 1],
    color="blue",
    marker="o",
    label="Projected Data (Lossless)",
)
axes[1].scatter(
    X_hat,
    np.zeros_like(X_hat),
    color="red",
    marker="x",
    label="1-dimensional PCA Data",
)
axes[1].set_title("Lossless and PCA Transformed Data")
axes[1].set_xlabel("u1")
axes[1].set_ylabel("u2")
axes[1].legend()

plt.tight_layout()
plt.show()
