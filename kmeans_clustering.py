# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 16:14:05 2026

@author: unangsunarya
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Memuat dataset Iris
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

print("Lima data pertama:")
print(X.head())

# 2. Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Membuat model K-Means
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# 4. Melakukan clustering tanpa label
clusters = kmeans.fit_predict(X_scaled)

# 5. Menambahkan hasil cluster
X["Cluster"] = clusters

print("\nJumlah data pada setiap cluster:")
print(X["Cluster"].value_counts().sort_index())

print("\nContoh hasil clustering:")
print(X.head(10))

# 6. Evaluasi menggunakan Silhouette Score
from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, clusters)
print("\nSilhouette Score:", round(score, 3))

# -1  Score  1

# score=1 benar2 terpisah dengan jelas
# score=0 data tumpang tindih
# score -1= missintrepediksi
#-----------------------------------------------------------


# 7. Reduksi dimensi menjadi 2D dengan PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 8. Visualisasi hasil clustering
plt.figure(figsize=(8, 6))

for cluster_id in np.unique(clusters):
    plt.scatter(
        X_pca[clusters == cluster_id, 0],
        X_pca[clusters == cluster_id, 1],
        label=f"Cluster {cluster_id}",
        s=60
    )

plt.title("Unsupervised Learning - K-Means Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
