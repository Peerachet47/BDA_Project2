# Group Members:
# Chalermchai Nichee (ID: 6531501015)
# Peerachet Khanitson (ID: 6531501092)
# Wisan Kittisaret (ID: 6531501197)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# Set Streamlit page configuration
st.set_page_config(page_title="Segmentation K-Means App", layout="centered")

# Title
st.title("Customer Segmentation using Pretrained K-Means Model")

@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    df = pd.read_excel(url)
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

@st.cache_resource
def load_model():
    with open("kmeans_customer_model.pkl", "rb") as file:
        return pickle.load(file)

df = load_data()
st.write("### Sample Data", df.head())

# Aggregate by customer
customer_df = df.groupby('CustomerID').agg({
    'InvoiceNo': 'nunique',
    'Quantity': 'sum',
    'TotalPrice': 'sum'
}).reset_index()
customer_df.columns = ['CustomerID', 'NumPurchases', 'TotalQuantity', 'TotalSpent']

# Normalize (ต้องเหมือนกับที่ใช้ฝึกโมเดล)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(customer_df.drop('CustomerID', axis=1))

# Load pretrained model
kmeans = load_model()
customer_df['Cluster'] = kmeans.predict(X_scaled)

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(customer_df['TotalQuantity'], customer_df['TotalSpent'],
                     c=customer_df['Cluster'], cmap='rainbow')
plt.xlabel('Total Quantity')
plt.ylabel('Total Spent')
plt.title('Customer Segments (Pretrained Model)')
st.pyplot(fig)

st.write("### Clustered Data", customer_df.head())
