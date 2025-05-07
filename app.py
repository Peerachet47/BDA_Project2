# Group Members:
# Chalermchai Nichee (ID: 6531501015)
# Peerachet Khanitson (ID: 6531501092)
# Wisan Kittisaret (ID: 6531501197)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Customer Segmentation Viewer", layout="centered")
st.title("📊 Customer Segmentation Viewer (Using Real Data)")

# Load model
@st.cache_resource
def load_model():
    with open("kmeans_customer_model.pkl", "rb") as f:
        return pickle.load(f)

# Load real customer data
@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

# Load resources
model = load_model()
df = load_data()

# Get feature count and number of clusters
n_features = model.cluster_centers_.shape[1]
num_clusters = model.n_clusters

# Select features to plot
feature_x = "Age"
feature_y = "Income"

# Ensure Cluster column exists
if "Cluster" not in df.columns:
    df["Cluster"] = model.predict(df.iloc[:, :n_features])

# Sidebar for cluster selection
st.sidebar.header("🔧 Options")
selected_cluster = st.sidebar.selectbox("Highlight Cluster", sorted(df["Cluster"].unique()))

# Set color: highlight selected cluster
colors = df["Cluster"].apply(lambda c: 'orange' if c == selected_cluster else 'gray')

# Plot
fig, ax = plt.subplots()
ax.scatter(df[feature_x], df[feature_y], c=colors, alpha=0.7, s=60, label="Customers")
centers = model.cluster_centers_

# Plot centroids (only first two features shown)
ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centroids')
ax.set_xlabel(feature_x)
ax.set_ylabel(feature_y)
ax.set_title(f"Customer Segmentation - Highlighting Cluster {selected_cluster}")
ax.legend()
st.pyplot(fig)

# Summary table
st.subheader("📋 Clustered Data Preview")
st.dataframe(df[df["Cluster"] == selected_cluster].head(10))

# Show warning if model has more than 2 features
if n_features > 2:
    st.warning(f"Note: Your model was trained on {n_features} features, but this plot only shows 2 (Age vs Income).")
