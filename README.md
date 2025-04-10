# 🛍️ Mall Customer Segmentation

This project performs **unsupervised machine learning (clustering)** to segment customers of a mall based on their spending behavior and demographics. The goal is to identify distinct groups to help improve marketing strategies and customer service.

---

## 📁 Project Structure

Mall-Customer-Segmentation-main/ ├── README.md # Project documentation ├── Unsupervised_Clustering_Solution.ipynb # Jupyter notebook for EDA and clustering ├── kmeans_model.pkl # Trained KMeans clustering model ├── mall_customers.csv # Dataset with customer information ├── mall_segment.py # Script for loading model and predicting segments ├── scaler.pkl # Pre-fitted scaler for preprocessing input data └── requirements.txt # Python dependencies

## 📊 Dataset

The dataset `mall_customers.csv` includes the following features:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1–100)

---

## 🧠 Techniques Used

- Data Preprocessing & Feature Scaling
- Elbow Method to find optimal `k`
- K-Means Clustering
- Cluster Visualization with Matplotlib/Seaborn
- Model saving and reuse with `pickle`

---

## 🛠️ How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt

📈 Results
Example output:

5 distinct customer segments identified

Clusters based on income and spending patterns

Visualizations highlight segmentation insights for marketing use

Streamlit app - https://dq293ur32.streamlit.app/

🤝 Acknowledgements Thanks to all contributors and dataset providers. This is a learning project for educational and demonstrative purposes.

🔗 Author: Arasu Ragupathi 📧 Contact: arasuragu23@gmail.com 🌟 GitHub: https://github.com/arasuu

