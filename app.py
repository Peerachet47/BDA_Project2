import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation with Pretrained KMeans Model")

@st.cache_data
def load_data():
    return pd.read_csv("clustered_customers.csv")

@st.cache_resource
def load_model():
    with open("kmeans_customer_model.pkl", "rb") as file:
        return pickle.load(file)

df = load_data()
model = load_model()

# Filter cluster
cluster_list = sorted(df["Cluster"].unique())
selected = st.sidebar.selectbox("Select Cluster", cluster_list)

st.subheader(f"📊 Summary of Cluster {selected}")
st.dataframe(df[df["Cluster"] == selected].describe())

# PCA Plot
st.subheader("🗺 PCA Scatter Plot")
fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", alpha=0.7)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Customer Clusters (PCA)")
plt.colorbar(scatter, ax=ax)
st.pyplot(fig)
