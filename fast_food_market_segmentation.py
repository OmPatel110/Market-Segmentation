import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

df = pd.read_csv("mcdonalds.csv")

df.head()

df.shape

df.info()

MD_x = df.iloc[:, :11].map(lambda x: 1 if x == "Yes" else 0)

# Calculate column means and round to 2 decimal places
column_means = MD_x.mean().round(2)

print(column_means)

# Perform PCA
pca = PCA()
MD_pca = pca.fit(MD_x)

# Standard deviations of the principal components
std_devs = np.sqrt(pca.explained_variance_)

# Explained variance ratio and cumulative variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Create a summary DataFrame
summary_df = pd.DataFrame({
    "Standard Deviation": std_devs,
    "Explained Variance Ratio": explained_variance_ratio,
    "Cumulative Variance": cumulative_variance
})

print(summary_df)

# Assuming MD_pca is the PCA object and MD_x is the data
pca_components = pca.transform(MD_x)  # Project data onto principal components

# Scatter plot of the first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], color='grey', alpha=0.7)
plt.title("PCA Projection")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)

# Annotate axes projections (like projAxes)
for i, vector in enumerate(pca.components_.T):
    plt.arrow(0, 0, vector[0]*max(pca_components[:, 0]), vector[1]*max(pca_components[:, 1]),
              color='red', alpha=0.7, head_width=0.02, length_includes_head=True)
    plt.text(vector[0]*max(pca_components[:, 0]) * 1.1,
             vector[1]*max(pca_components[:, 1]) * 1.1,
             f"Var{i+1}", color='red')

plt.show()



# Compute WCSS for different numbers of clusters
wcss = []
max_clusters = 10  # Adjust as needed
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(pca_components)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS

# Plot WCSS (Elbow Method)
plt.figure(figsize=(8, 6))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--', color='b')
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.xticks(range(1, max_clusters + 1))
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Perform k-means clustering on the PCA-transformed data
kmeans = KMeans(n_clusters=3, random_state=42)  # Change n_clusters as needed
kmeans_labels = kmeans.fit_predict(pca_components)

# Scatter plot of the first two principal components with k-means cluster labels
plt.figure(figsize=(8, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering on PCA Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.show()
