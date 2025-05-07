# Group Members:
# Chalermchai Nichee (ID: 6531501015)
# Peerachet Khanitson (ID: 6531501092)
# Wisan Kittisaret (ID: 6531501197)

import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Load model
with open('kmeans_customer_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

# Get number of clusters and features from the model
num_clusters = loaded_model.n_clusters
n_features = loaded_model.cluster_centers_.shape[1]

# Set Streamlit page configuration
st.set_page_config(page_title="K-Means Clustering Viewer", layout="centered")
st.title("📊 K-Means Clustering Visualizer")

# Sidebar options
st.sidebar.header("🔧 Options")
selected_cluster = st.sidebar.slider("Highlight Cluster", min_value=0, max_value=num_clusters - 1, value=0)

# Generate synthetic data with correct number of features
X, _ = make_blobs(n_samples=300, centers=num_clusters, n_features=n_features, cluster_std=0.60, random_state=0)

# Predict clusters
y_kmeans = loaded_model.predict(X)

# Build color list for highlighting selected cluster
colors = ['orange' if cluster == selected_cluster else 'gray' for cluster in y_kmeans]

# Plot (only first 2 features shown)
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=colors, s=50, alpha=0.7, label='Data Points')
ax.scatter(loaded_model.cluster_centers_[:, 0], loaded_model.cluster_centers_[:, 1],
           s=300, c='red', marker='X', label='Centroids')

ax.set_xlabel("Age")
ax.set_ylabel("Spending Score")
ax.set_title(f"Cluster View: Highlighting Cluster {selected_cluster}")
ax.legend()
st.pyplot(fig)

# Optional note if the model has more than 2 features
if n_features > 2:
    st.warning(f"Note: Your model was trained on {n_features} features. Only the first 2 features are visualized here.")
