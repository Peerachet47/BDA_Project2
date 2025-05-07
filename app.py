# Group Members:
# Chalermchai Nichee (ID: 6531501015)
# Peerachet Khanitson (ID: 6531501092)
# Wisan Kittisaret (ID: 6531501197)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("🧠 Customer Segmentation with KMeans and PCA")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/clustered_customers.csv")

# Load pretrained KMeans model
@st.cache_resource
def load_model():
    with open("kmeans_customer_model.pkl", "rb") as file:
        return pickle.load(file)

df = load_data()
model = load_model()

# Sidebar: select cluster
st.sidebar.header("Cluster Filter")
cluster_list = sorted(df["Cluster"].unique())
selected = st.sidebar.selectbox("Choose Cluster", cluster_list)

# Filtered summary
st.subheader(f"📊 Summary of Cluster {selected}")
st.dataframe(df[df["Cluster"] == selected].describe())

# Scatter plot with PCA
st.subheader("🗺 PCA Scatter Plot")
fig, ax = plt.subplots()
scatter = ax.scatter(df["PCA1"], df["PCA2"], c=df["Cluster"], cmap="viridis", alpha=0.6)
ax.set_xlabel("PCA1")
ax.set_ylabel("PCA2")
ax.set_title("Customer Clusters (PCA Projection)")
plt.colorbar(scatter, ax=ax, label="Cluster")
st.pyplot(fig)

# Show full table if needed
with st.expander("🔍 View All Clustered Data"):
    st.dataframe(df)
