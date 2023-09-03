# PCA With Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Get the mean
        self.mean = np.mean(X, axis=0)
        # Subtract the mean from the X
        X = X - self.mean
        # Calculate covariance matrix
        cov = np.cov(X.T)
        # Get the eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T  # As it comes as a column vector
        idxs = np.argsort(eigenvalues)[::-1] # sort the eigenvalues
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        # Store the first n eigenvectors
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)





data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()
