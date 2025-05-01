import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from bioinformatics_clusting import BioinformaticsClustering

# Set random seed for reproducibility
np.random.seed(42)

# Create a simulated gene expression dataset (100 samples x 500 genes)
# With 4 underlying clusters
n_samples = 100
n_features = 500
n_clusters = 4

# Generate synthetic data with known clusters
X, y_true = make_blobs(n_samples=n_samples, n_features=n_features, 
                      centers=n_clusters, cluster_std=1.5, 
                      random_state=42)

# Create sample and gene names
sample_names = [f'Sample_{i}' for i in range(n_samples)]
gene_names = [f'Gene_{i}' for i in range(n_features)]

# Convert to pandas DataFrame
df = pd.DataFrame(X, index=sample_names, columns=gene_names)

# Add some random noise to make it more realistic
noise = np.random.normal(0, 0.5, df.shape)
df = df + noise

# Initialize clustering object with k-means method
kmeans_clustering = BioinformaticsClustering(method='kmeans')

# Find optimal number of clusters
optimal_clusters = kmeans_clustering.determine_optimal_clusters(df, max_clusters=10)
print(f"Recommended number of clusters: {optimal_clusters['recommendations']['overall']}")

# Fit the model with the recommended number of clusters
n_clusters = optimal_clusters['recommendations']['overall']
kmeans_clustering.fit(df, n_clusters=n_clusters)

# Evaluate clustering performance
metrics = kmeans_clustering.evaluate_clusters(df, y_true)
print(f"Clustering metrics: {metrics}")

# Plot the results
plt.figure(figsize=(18, 12))

# Plot 1: Cluster visualization
plt.subplot(221)
fig_clusters = kmeans_clustering.plot_clusters(df)

# Plot 2: Silhouette analysis
plt.subplot(222)
fig_silhouette = kmeans_clustering.plot_silhouette(df)

# Plot 3: Heatmap
plt.subplot(223)
fig_heatmap = kmeans_clustering.plot_heatmap(df, n_genes=50)

# Plot 4: Cluster validation metrics
plt.subplot(224)
fig_validation = kmeans_clustering.plot_cluster_validation(df, max_clusters=10)

plt.tight_layout()
plt.show()

# Try another clustering method (hierarchical)
hier_clustering = BioinformaticsClustering(method='hierarchical')
hier_clustering.fit(df, n_clusters=n_clusters, linkage='ward')

# Compare results
print("Hierarchical clustering metrics:")
print(hier_clustering.evaluate_clusters(df, y_true))

# Plot dendrogram
fig_dendrogram = hier_clustering.plot_dendrogram(df, truncate_mode='level', p=3)
plt.show()

# For network-based analysis, try WGCNA (if dependencies available)
try:
    wgcna_clustering = BioinformaticsClustering(method='wgcna')
    wgcna_clustering.fit(df, power=6, minModuleSize=20)
    
    # Plot network
    fig_network = wgcna_clustering.plot_wgcna_network()
    plt.show()
except ImportError:
    print("WGCNA dependencies not available. Skipping WGCNA analysis.")

