import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("kmeans_model.pkl")

# Set page config
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("🧠 Customer Segmentation using KMeans")
st.write("Enter a customer's details to predict their segment and visualize their position in the customer base.")

# Sidebar inputs
st.sidebar.header("📝 Input Customer Data")
income = st.sidebar.slider("Annual Income (k$)", 0, 150, 60)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Predict cluster
input_data = np.array([[income, score]])
cluster = model.predict(input_data)[0]

# Display prediction
st.subheader(f"🏷️ Predicted Segment: Cluster {cluster}")

# Optional insights
cluster_names = {
    0: "💼 Budget-Conscious",
    1: "💎 VIP Spenders",
    2: "📉 Low Value",
    3: "🛍️ Mid-Tier Spenders",
    4: "📊 Balanced Customers"
}
st.markdown(f"**Segment Insight:** {cluster_names.get(cluster, 'Customer Segment')}")

# Visualize clusters with user's point
st.subheader("📊 Cluster Visualization")

# Generate example dataset (for demo or if you have df saved, you can load it)
# Replace with your real data if available
# df = pd.read_csv("mall_customers.csv")
# For now, simulate 200 points per cluster center
df = pd.DataFrame(model.cluster_centers_, columns=["Annual_Income", "Spending_Score"])
df["Cluster"] = df.index

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x="Annual_Income", y="Spending_Score", hue="Cluster", palette="Set2", data=df, s=100)
plt.scatter(income, score, color="red", s=200, edgecolor="black", label="New Customer")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Clusters")
plt.legend()
st.pyplot(plt)

# Footer
st.markdown("---")
st.caption("Built with 💙 using KMeans and Streamlit")
