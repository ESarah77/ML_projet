from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# generate data for linear regression
X, y = make_regression(n_samples=1000, n_features=1, noise=5.0, random_state=42)

# visualise the data
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, alpha=0.5)
plt.title("Generated Linear Regression Data")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.savefig('gen_linear_reg.pdf')
plt.show()
