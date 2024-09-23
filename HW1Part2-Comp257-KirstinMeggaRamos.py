# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:22:31 2024

@author: Kirstin Ramos
COMP 257 - Homework 1 Part 2
"""

#Generate Swiss roll dataset. [5 points]
from sklearn.datasets import make_swiss_roll

X_swiss, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=82)

#Plot the resulting generated Swiss roll dataset. [2 points]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the Swiss roll dataset with color mapped to t
ax.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=t, cmap="inferno", s=10)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

#Use Kernel PCA (kPCA) with linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points). [6 points]
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# Kernel PCA with linear kernel
kpca_linear = KernelPCA(kernel="linear", n_components=2)
X_kpca_linear = kpca_linear.fit_transform(X_swiss)

# Kernel PCA with RBF kernel
kpca_rbf = KernelPCA(kernel="rbf", n_components=2, gamma=0.04)
X_kpca_rbf = kpca_rbf.fit_transform(X_swiss)

# Kernel PCA with sigmoid kernel
kpca_sigmoid = KernelPCA(kernel="sigmoid", n_components=2, gamma=0.001, coef0=1)
X_kpca_sigmoid = kpca_sigmoid.fit_transform(X_swiss)

#Plot the kPCA results of applying the linear kernel (2 points), a RBF kernel (2 points), and a sigmoid kernel (2 points) 
#from (3). Explain and compare the results [6 points]

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Linear kernel PCA plot
ax1.scatter(X_kpca_linear[:, 0], X_kpca_linear[:, 1], c=t, cmap='inferno', s=10)
ax1.set_title("Kernel PCA with Linear Kernel")

# RBF kernel PCA plot
ax2.scatter(X_kpca_rbf[:, 0], X_kpca_rbf[:, 1], c=t, cmap='inferno', s=10)
ax2.set_title("Kernel PCA with RBF Kernel")

# Sigmoid kernel PCA plot
ax3.scatter(X_kpca_sigmoid[:, 0], X_kpca_sigmoid[:, 1], c=t, cmap='inferno', s=10)
ax3.set_title("Kernel PCA with Sigmoid Kernel")

plt.show()

#Using kPCA and a kernel of your choice, apply Logistic Regression for classification. 
#Use GridSearchCV to find the best kernel and gamma value for kPCA in order 
#to get the best classification accuracy at the end of the pipeline. 
#Print out best parameters found by GridSearchCV. [14 points]

from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Convert this to a binary classification problem by thresholding t
y = (t > 6).astype(int)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_swiss, y, test_size=0.2, random_state=82)

# Standardize the data (important for many algorithms, including logistic regression)
scaler = StandardScaler()

# Create a pipeline with KernelPCA and LogisticRegression
pipeline = Pipeline([
    ('scaler', scaler),
    ('kpca', KernelPCA(n_components=2)),
    ('log_reg', LogisticRegression(solver='lbfgs'))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'kpca__kernel': ['linear', 'rbf', 'sigmoid'],   # Kernels to try
    'kpca__gamma': [0.01, 0.1, 1, 5, 10]            # Gamma values to try
}

# Apply GridSearchCV to find the best kernel and gamma
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
print("Best parameters found by GridSearchCV:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

# Test the best model on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("Test accuracy with the best model:", test_accuracy)

#Plot the results from using GridSearchCV in (5). [2 points]
import numpy as np

# Extract the results from the GridSearchCV object
results = grid_search.cv_results_

# Get the parameter combinations and their mean test scores
mean_test_scores = results['mean_test_score']
kernels = results['param_kpca__kernel'].data
gammas = results['param_kpca__gamma'].data

# Reshape scores to create a 2D grid for each kernel
gammas_unique = sorted(list(set(gammas)))
kernel_types = sorted(list(set(kernels)))

# Prepare a grid for each kernel type
fig, ax = plt.subplots(1, len(kernel_types), figsize=(15, 5))

for i, kernel in enumerate(kernel_types):
    scores = np.array([mean_test_scores[j] for j in range(len(kernels)) if kernels[j] == kernel])
    ax[i].plot(gammas_unique, scores, marker='o')
    ax[i].set_title(f'Kernel: {kernel}')
    ax[i].set_xlabel('Gamma')
    ax[i].set_ylabel('Mean Accuracy')
    ax[i].grid(True)

plt.tight_layout()
plt.show()



