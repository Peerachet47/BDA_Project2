"""
Created on Wed May 07 14:28:26 2025

@author: peerachet
"""
# Group Members:
# Chalermchai Nichee (ID: 6531501015)
# Peerachet Khanitson (ID: 6531501092)
# Wisan Kittisaret (ID: 6531501197)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.title("Retail Customer Segmentation using K-Means")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()
st.write("### Sample Data", df.head())

# Aggregate by customer
customer_df = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()
customer_df.columns = ['CustomerID', 'NumPurchases', 'TotalQuantity', 'TotalSpent']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df.drop('CustomerID', axis=1))

# K-Means
k = st.slider("Select number of clusters (K):", 2, 10, 4)
kmeans = KMeans(n_clusters=k, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(customer_df['TotalQuantity'], customer_df['TotalSpent'],
                     c=customer_df['Cluster'], cmap='rainbow')
plt.xlabel('Total Quantity')
plt.ylabel('Total Spent')
plt.title('Customer Segments')
st.pyplot(fig)

st.write("### Clustered Data", customer_df.head())
