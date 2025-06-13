import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
public_df = pd.read_csv("public_data.csv", header=None)
public_df = public_df.drop(index=0).reset_index(drop=True)
public_df.columns = ['id', 'S1', 'S2', 'S3', 'S4']

# 1. Original feature distributions
fig1, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(['S1', 'S2', 'S3', 'S4']):
    sns.histplot(public_df[col].astype(float), ax=axes[i], bins=100, kde=True)
    axes[i].set_title(f'Original {col}')
fig1.tight_layout()
fig1.savefig("original_distribution.png")

# 2. Standardized feature distributions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(public_df[['S1', 'S2', 'S3', 'S4']].astype(float))
df_scaled = pd.DataFrame(X_scaled, columns=['S1', 'S2', 'S3', 'S4'])

fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, col in enumerate(['S1', 'S2', 'S3', 'S4']):
    sns.histplot(df_scaled[col], ax=axes[i], bins=100, kde=True)
    axes[i].set_title(f'Standardized {col}')
fig2.tight_layout()
fig2.savefig("scaled_distribution.png")

# 3. Inter-dimensional scatter plots
fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
pairs = [('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4')]
for ax, (x, y) in zip(axes, pairs):
    ax.scatter(public_df[x].astype(float), public_df[y].astype(float), s=1, alpha=0.5)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'{x} vs {y}')
fig3.tight_layout()
fig3.savefig("inter_dim_scatter.png")


# Load and clean public dataset
public_df = pd.read_csv("public_data.csv", header=None)
public_df = public_df.drop(index=0).reset_index(drop=True)
public_df.columns = ['id', 'S1', 'S2', 'S3', 'S4']

# Extract and scale features
X_pub = public_df[['S1', 'S2', 'S3', 'S4']].astype(float).values
scaler = StandardScaler()
X_pub_scaled = scaler.fit_transform(X_pub)

# Clustering again (to recover predicted labels)
kmeans = KMeans(n_clusters=15, n_init=10, random_state=42)
y_pred = kmeans.fit_predict(X_pub_scaled)
public_df["cluster"] = y_pred

# Scatter plots colored by predicted cluster
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
pairs = [('S1', 'S2'), ('S2', 'S3'), ('S3', 'S4')]
for ax, (x, y) in zip(axes, pairs):
    sns.scatterplot(data=public_df, x=x, y=y, hue='cluster', palette='tab20', s=10, ax=ax, legend=False)
    ax.set_title(f'{x} vs {y} by cluster')

fig.tight_layout()
plt.savefig("public_clustering_visualization.png")
