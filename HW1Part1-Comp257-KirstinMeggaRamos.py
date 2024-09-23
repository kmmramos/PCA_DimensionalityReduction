# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:26:59 2024

@author: Kirstin Ramos
COMP 257 - Homework 1 Part 1
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import fetch_openml
import tensorflow as tf

import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

#Retrieve and load the mnist_784 dataset of 70,000 instances. [5 points]
mnist = fetch_openml('mnist_784', as_frame=False)
X_train, y_train =  mnist.data[:70_000], mnist.target[:70_000]
X_test, y_test = mnist.data[70_000:], mnist.target[70_000:]

# Convert labels to integers
labels = y_train.astype(int)

# Display each digit
def display_digits(indices):
    # Load MNIST dataset from tensorflow
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Create subplots with 1 row and len(indices) columns
    fig, axes = plt.subplots(1, len(indices), figsize=(len(indices)*2, 2))
    
    # Loop through each index and corresponding subplot axis
    for i, index in enumerate(indices):
        image = X_train[index]
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'{y_train[index]}')
        axes[i].axis('off')  # Turn off axis for cleaner look
    
    plt.show()

# Display the first 10 digits
display_digits(range(10))

# Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio

# Flatten the images from 28x28 to 784 (as PCA works on 2D arrays)
X_train_flattened = X_train.reshape(X_train.shape[0], -1)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_train_flattened)

# Output the explained variance ratio of the first 2 principal components
explained_variance_ratio = pca.explained_variance_ratio_

print("Explained Variance Ratio of 1st Component:", explained_variance_ratio[0])
print("Explained Variance Ratio of 2nd Component:", explained_variance_ratio[1])

#Plot the projections of the 1st and 2nd principal component onto a 2D hyperplane. [5 points]
plt.figure(figsize=(10, 6))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', s=1, alpha=0.5)
plt.colorbar(scatter, label='Digit Label')
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.title('PCA Projection of MNIST Digits')
plt.show()

#Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions. [10 points]
from sklearn.decomposition import IncrementalPCA

# Initialize IncrementalPCA to reduce to 154 dimensions
ipca = IncrementalPCA(n_components=154)

# Fit IncrementalPCA in batches
batch_size = 10000
for start in range(0, X_train_flattened.shape[0], batch_size):
    end = min(start + batch_size, X_train_flattened.shape[0])
    ipca.partial_fit(X_train_flattened[start:end])

# Transform the data to 154 dimensions
X_train_reduced = ipca.transform(X_train_flattened)

# Output the shape of the reduced data
print("Original data shape:", X_train_flattened.shape)
print("Reduced data shape:", X_train_reduced.shape)

from sklearn.decomposition import IncrementalPCA
import numpy as np

# Number of batches
n_batches = 100

# Initialize IncrementalPCA to reduce to 154 dimensions
inc_pca = IncrementalPCA(n_components=154)

# Split the data into n_batches and apply partial_fit
for X_batch in np.array_split(X_train_flattened, n_batches):
    inc_pca.partial_fit(X_batch)

# Transform the entire dataset after fitting
X_train_reduced = inc_pca.transform(X_train_flattened)

# Output the shape of the reduced data
print("Original data shape:", X_train_flattened.shape)
print("Reduced data shape:", X_train_reduced.shape)

# Use the already fitted IncrementalPCA (ipca) to reconstruct images
X_train_reconstructed = inc_pca.inverse_transform(X_train_reduced)

# Check the size of the reconstructed data
print(f"Reconstructed data shape: {X_train_reconstructed.shape}")

# Ensure the correct number of samples
n_samples = X_train_flattened.shape[0]

# Reshape the reconstructed images back to (n_samples, 28, 28)
X_train_reconstructed_reshaped = X_train_reconstructed.reshape(n_samples, 28, 28)

# Function to display original and reconstructed images
def display_images(original_images, reconstructed_images, indices):
    fig, axes = plt.subplots(2, len(indices), figsize=(len(indices)*2, 4))
    
    for i, index in enumerate(indices):
        # Display original images
        axes[0, i].imshow(original_images[index].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f'Original {labels[index]}')
        axes[0, i].axis('off')
        
        # Display reconstructed images
        axes[1, i].imshow(reconstructed_images[index], cmap='gray')
        axes[1, i].set_title(f'Reconstructed {labels[index]}')
        axes[1, i].axis('off')
    
    plt.show()

# Display the first 10 original and reconstructed digits
display_images(X_train_flattened, X_train_reconstructed_reshaped, range(10))

