import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# App layout
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("🧠 Customer Segmentation with KMeans")
st.write("Enter income and spending score to see which customer segment they fall into.")

# Sidebar inputs
income = st.sidebar.slider("Annual Income (k$)", 0, 150, 60)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Preprocess input
input_data = np.array([[income, score]])
input_scaled = scaler.transform(input_data)

# Predict cluster
cluster = model.predict(input_scaled)[0]
st.subheader(f"🏷️ Predicted Segment: Cluster {cluster}")

# Visualize
st.subheader("📊 Cluster Visualization (Centers Only)")

centers = scaler.inverse_transform(model.cluster_centers_)
centers = np.round(centers, 2)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=centers[:, 0], y=centers[:, 1], hue=[f'Cluster {i}' for i in range(len(centers))], palette="Set2", s=200, legend='full')
plt.scatter(income, score, color='red', s=200, edgecolor='black', label='Your Input')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Cluster Centers and Your Customer")
plt.legend()
st.pyplot(plt)

# Footer
st.markdown("---")
st.caption("Built with 🧠 KMeans + Streamlit")
