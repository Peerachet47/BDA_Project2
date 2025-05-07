# app.py

import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.title("🧠 Customer Segmentation using KMeans")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

# Load model
@st.cache_resource
def load_model():
    with open("kmeans_customer_model.pkl", "rb") as file:
        return pickle.load(file)

df = load_data()
kmeans_model = load_model()

# Sidebar - Cluster filter
clusters = sorted(df["Cluster"].unique())
selected_cluster = st.sidebar.selectbox("Select Cluster", clusters)

filtered = df[df["Cluster"] == selected_cluster]

st.subheader(f"📊 Cluster {selected_cluster} Summary")
st.write(filtered.describe())

# Scatter plot (PCA1 vs PCA2)
st.subheader("🗺 PCA Scatter Plot of All Clusters")
fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", alpha=0.6)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Customer Clusters by PCA")
plt.colorbar(scatter, ax=ax, label="Cluster")
st.pyplot(fig)
