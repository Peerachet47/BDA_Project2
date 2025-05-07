import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation using KMeans")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

# Load Model
@st.cache_resource
def load_model():
    with open("kmeans_customer_model.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
model = load_model()

# Sidebar filter
cluster_option = st.sidebar.selectbox("Select Cluster", sorted(df["Cluster"].unique()))
filtered_df = df[df["Cluster"] == cluster_option]

# Display summary
st.subheader(f"📊 Summary for Cluster {cluster_option}")
st.dataframe(filtered_df.describe())

# PCA scatter plot
st.subheader("🗺 PCA Scatter Plot of Clusters")
fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", alpha=0.6)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("PCA-based Customer Clusters")
plt.colorbar(scatter, ax=ax, label="Cluster")
st.pyplot(fig)
