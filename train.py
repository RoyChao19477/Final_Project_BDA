import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load datasets
public_df = pd.read_csv("public_data.csv", header=None)
private_df = pd.read_csv("private_data.csv", header=None, low_memory=False)

# Drop the header row and ID column
public_df = public_df.drop(index=0).reset_index(drop=True)
private_df = private_df.drop(index=0).reset_index(drop=True)

# Extract features only (remove ID and label columns)
X_pub = public_df.iloc[:, 1:5].astype(float).values  # 4 features
X_priv = private_df.iloc[:, 1:7].astype(float).values  # 6 features

# Scale each dataset separately
scaler_pub = StandardScaler()
scaler_priv = StandardScaler()
X_pub_scaled = scaler_pub.fit_transform(X_pub)
X_priv_scaled = scaler_priv.fit_transform(X_priv)

# PCA to reduce dimensionality (optional but helps stability)
#pca_pub = PCA(n_components=3)
#pca_priv = PCA(n_components=3)
#X_pub_pca = pca_pub.fit_transform(X_pub_scaled)
#X_priv_pca = pca_priv.fit_transform(X_priv_scaled)

# Clustering
kmeans_pub = KMeans(n_clusters=15, n_init=10, random_state=42)
kmeans_priv = KMeans(n_clusters=23, n_init=10, random_state=42)

#y_pub_pred = kmeans_pub.fit_predict(X_pub_pca)
#y_priv_pred = kmeans_priv.fit_predict(X_priv_pca)
y_pub_pred = kmeans_pub.fit_predict(X_pub_scaled)
y_priv_pred = kmeans_priv.fit_predict(X_priv_scaled)

# Save results
pd.DataFrame({
        "id": range(len(y_pub_pred)),
            "label": y_pub_pred
            }).to_csv("public_submission.csv", index=False)

pd.DataFrame({
        "id": range(len(y_priv_pred)),
            "label": y_priv_pred
            }).to_csv("private_submission.csv", index=False)

print("Done. Output files: public_submission.csv, private_submission.csv")
