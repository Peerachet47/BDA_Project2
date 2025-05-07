# Group Members:
# Chalermchai Nichee (ID: 6531501015)
# Peerachet Khanitson (ID: 6531501092)
# Wisan Kittisaret (ID: 6531501197)

import joblib
import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.title("Retail Customer Segmentation using Pretrained K-Means Model")

@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")
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

# Load pretrained model
model_path = "k-mean_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ k-mean_model.pkl not found. Please train the model first.")
    st.stop()

kmeans = joblib.load(model_path)
customer_df['Cluster'] = kmeans.predict(X_scaled)

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(customer_df['TotalQuantity'], customer_df['TotalSpent'],
                     c=customer_df['Cluster'], cmap='rainbow')
plt.xlabel('Total Quantity')
plt.ylabel('Total Spent')
plt.title('Customer Segments')
st.pyplot(fig)

st.write("### Clustered Data", customer_df.head())