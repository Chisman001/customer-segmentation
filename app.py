import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
  page_title="Customer Segmentation Dashboard",
  page_icon="🛍️",
  layout="wide"
)

st.image("images/banner.png", use_container_width=True)


@st.cache_resource
def load_models():
    model_dir = Path("models")
    scaler_path = model_dir / "scaler.pkl"
    kmeans_path = model_dir / "kmeans_model.pkl"

    if not scaler_path.exists() or not kmeans_path.exists():
        return None

    return joblib.load(scaler_path), joblib.load(kmeans_path)


df = pd.read_csv("data/store_customers.csv")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Dataset",
        "EDA",
        "Correlation Analysis",
        "Customer Segmentation",
        "Business Insights"
    ]
)

if page == "Overview":
    st.title("🛍️ Customer Segmentation Dashboard")

    st.markdown("""
    ### 📖 Project Overview

    This dashboard presents a customer segmentation analysis using the **K-Means Clustering** algorithm. The objective is to identify groups of customers with similar purchasing behaviors based on their **Annual Income** and **Spending Score**.

    The analysis includes exploratory data analysis (EDA), customer segmentation, cluster visualization, and business recommendations to support data-driven marketing strategies.
    """)

    st.subheader("📊 Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers", len(df))
    col2.metric("Average Income", round(df["Annual Income (k$)"].mean(), 2))
    col3.metric("Average Spending Score", round(df["Spending Score (1-100)"].mean(), 2))
    col4.metric('Average Age', round(df["Age"].mean(), 2))

elif page == "Dataset":
    st.header("📋 Dataset Preview")

    st.write(
        "This dataset contains customer demographic information and spending behaviour used for customer segmentation."
    )

    st.dataframe(df.head(10))

    rows = st.slider("Select number of rows", 5, 50, 10)

    st.dataframe(df.head(rows))

    st.header("📊 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Rows:**", df.shape[0])
        st.write("**Columns:**", df.shape[1])

    with col2:
        st.write("**Missing Values:**", df.isnull().sum().sum())
        st.write("**Duplicate Rows:**", df.duplicated().sum())
elif page == "EDA":
    st.header("📈 Exploratory Data Analysis (EDA)")

    chart = st.selectbox(
        "Choose a chart",
        [
            "Age Distribution",
            "Income Distribution",
            "Spending Score Distribution",
            "Correlation Heatmap",
            "Scatter Plot"
        ]
    )

    if chart == "Age Distribution":
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df["Age"], bins=20)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
    elif chart == "Income Distribution":
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df["Annual Income (k$)"], bins=20)
        ax.set_title("Annual Income Distribution")
        st.pyplot(fig)
    elif chart == "Spending Score Distribution":
        fig, ax = plt.subplots(figsize=(7,4))
        ax.hist(df["Spending Score (1-100)"], bins=20)
        ax.set_title("Spending Score Distribution")
        st.pyplot(fig)
    elif chart == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(
            df.select_dtypes(include="number").corr(),
            annot=True,
            cmap="coolwarm",
            ax=ax
        )
        st.pyplot(fig)
    elif chart == "Scatter Plot":
        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(
            df["Annual Income (k$)"],
            df["Spending Score (1-100)"]
        )
        ax.set_xlabel("Annual Income")
        ax.set_ylabel("Spending Score")
        st.pyplot(fig)
elif page == "Correlation Analysis":
    st.header("📉 Correlation Analysis")

    st.write("""
    Correlation analysis measures the strength and direction of the relationship
    between numerical variables. Correlation values range from **-1** to **1**:

    - **1** → Perfect positive relationship
    - **0** → No linear relationship
    - **-1** → Perfect negative relationship
    """)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        variable1 = st.selectbox("Select first variable", numeric_columns)

    with col2:
        variable2 = st.selectbox("Select second variable", numeric_columns)

    correlation = df[variable1].corr(df[variable2])

    st.metric("Correlation Coefficient", f"{correlation:.2f}")

    if correlation >= 0.7:
        st.success("Strong positive correlation")

    elif correlation >= 0.3:
        st.info("Moderate positive correlation")
    elif correlation >= -0.3:
        st.warning("Weak or no correlation")
    elif correlation >= -0.7:
        st.info("Moderate negative correlation")
    elif correlation == 1:
        st.success("Perfect positive correlation")
    elif correlation == -1:
        st.error("Perfect negative correlation")
    else:
        st.error("Strong negative correlation")
elif page == "Customer Segmentation":
    st.header("🤖 Customer Segmentation")

    st.write("""
    The K-Means clustering algorithm was used to segment customers into **four distinct groups**
    based on their **Annual Income** and **Spending Score**. Each colour in the scatter plot
    represents a different customer segment with similar purchasing behaviour.
    """)

    models = load_models()
    if models is None:
        st.error("Trained models not found. Run: python train.py")
        st.stop()

    scaler, kmeans = models

    df = df.dropna().copy()
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    X_scaled = scaler.transform(X)
    df["Cluster"] = kmeans.predict(X_scaled)

    fig, ax = plt.subplots(figsize=(9,6))

    ax.scatter(
        df["Annual Income (k$)"],
        df["Spending Score (1-100)"],
        c=df["Cluster"]
    )

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=300,
        c="red",
        label="Centroids"
    )

    ax.set_title("Customer Segments")
    ax.set_xlabel("Annual Income (k$)")
    ax.set_ylabel("Spending Score (1-100)")
    ax.legend()

    st.pyplot(fig)

    cluster_summary = df.groupby("Cluster")[
        ["Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"]
    ].mean()

    st.subheader("📊 Cluster Summary")

    st.dataframe(cluster_summary)

    cluster_counts = df["Cluster"].value_counts().sort_index()

    st.subheader("📈 Customers in Each Cluster")

    st.bar_chart(cluster_counts)
elif page == "Business Insights":
    st.subheader("💡 Key Business Insights")

    st.markdown("""
    - Young customers generally have higher spending scores.
    - High-income customers tend to spend more conservatively.
    - The largest customer segment consists of regular shoppers.
    - Businesses can tailor promotions to each customer segment to improve engagement.
    """)