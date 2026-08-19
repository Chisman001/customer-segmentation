import joblib
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv("data/store_customers.csv").dropna()
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
kmeans.fit(X_scaled)

joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
joblib.dump(kmeans, MODEL_DIR / "kmeans_model.pkl")

print(f"Saved models to {MODEL_DIR}/")
