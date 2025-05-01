# Bioinformatics Dimensionality Reducer

A comprehensive Python library for dimensionality reduction in bioinformatics data analysis. This tool provides an integrated framework for applying and visualizing various dimensionality reduction techniques with a focus on biological data interpretation.

## Features

- **Multiple reduction methods** including:
  - PCA (Principal Component Analysis)
  - LDA (Linear Discriminant Analysis)
  - t-SNE (t-Distributed Stochastic Neighbor Embedding)
  - UMAP (Uniform Manifold Approximation and Projection)
  - Autoencoder (Neural network-based reduction)

- **Built-in visualization** tools:
  - Reduced dimension plots with categorical coloring
  - Explained variance analysis
  - Feature loading/importance plots
  - Customizable plotting options

- **Bioinformatics-focused utilities**:
  - Component interpretation
  - Feature importance analysis
  - Model saving and loading

## Installation

### Prerequisites

This package requires Python 3.7+ and the following dependencies:

```bash
numpy>=1.19.0
pandas>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=0.23.0
tensorflow>=2.4.0
```

Optional dependencies:
```bash
umap-learn>=0.5.0  # For UMAP method
```

## Quick Start

```python
from bioinformatics_dimensionalityreducer import BioinformaticsDimensionalityReducer
import pandas as pd
import matplotlib.pyplot as plt

# Load your data
gene_expression = pd.read_csv('gene_expression_data.csv', index_col=0)
sample_groups = pd.read_csv('sample_metadata.csv', index_col=0)['condition']

# Initialize reducer with PCA
reducer = BioinformaticsDimensionalityReducer(method='pca', n_components=2)

# Fit and transform data
reduced_data = reducer.fit_transform(gene_expression)

# Visualize results
fig = reducer.plot_reduced_data(X_reduced=reduced_data, y=sample_groups)
plt.show()

# Analyze feature importance
reducer.plot_feature_loadings(n_features=15)
plt.show()
```

## Detailed Usage

### Available Methods

#### PCA (Principal Component Analysis)
```python
# Basic PCA
pca_reducer = BioinformaticsDimensionalityReducer(method='pca', n_components=2)
pca_result = pca_reducer.fit_transform(data)

# Analyze variance explained
pca_reducer.plot_explained_variance()

# View feature loadings
loadings_df = pca_reducer.get_feature_loadings()
```

#### LDA (Linear Discriminant Analysis)
```python
# LDA requires labeled data
lda_reducer = BioinformaticsDimensionalityReducer(method='lda', n_components=2)
lda_result = lda_reducer.fit_transform(data, labels)

# Plot discriminant feature loadings
lda_reducer.plot_feature_loadings(component_idx=0)  # First discriminant
```

#### t-SNE
```python
# t-SNE with custom parameters
tsne_reducer = BioinformaticsDimensionalityReducer(
    method='tsne', 
    n_components=2
)
tsne_result = tsne_reducer.fit_transform(
    data, 
    perplexity=30,
    learning_rate=200.0,
    n_iter=1000
)
```

#### UMAP
```python
# UMAP with default parameters
umap_reducer = BioinformaticsDimensionalityReducer(method='umap', n_components=2)
umap_result = umap_reducer.fit_transform(data)

# UMAP with custom parameters
umap_custom = BioinformaticsDimensionalityReducer(method='umap', n_components=2)
umap_custom_result = umap_custom.fit_transform(
    data,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)

# Supervised UMAP
umap_supervised = BioinformaticsDimensionalityReducer(method='umap', n_components=2)
umap_supervised_result = umap_supervised.fit_transform(data, labels)
```

#### Autoencoder
```python
# Autoencoder with custom architecture
ae_reducer = BioinformaticsDimensionalityReducer(method='autoencoder', n_components=2)
ae_result = ae_reducer.fit_transform(
    data,
    hidden_layers=[512, 256, 128],
    activation='relu',
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    dropout_rate=0.2
)
```

### Visualization Options

#### Basic Plot
```python
# Default visualization
reducer.plot_reduced_data(X_reduced=reduced_data)
```

#### Enhanced Plot with Categorical Coloring
```python
# Plot with sample groups
fig = reducer.plot_reduced_data(
    X_reduced=reduced_data,
    y=sample_groups,
    figsize=(12, 10),
    alpha=0.8,
    s=70,
    title='Gene Expression Clusters',
    cmap='viridis',
    add_legend=True,
    legend_title='Sample Group'
)
```

#### Feature Importance Visualization
```python
# Plot top 20 features for first component
reducer.plot_feature_loadings(
    n_features=20,
    component_idx=0,
    figsize=(14, 10),
    absolute=True
)
```

### Saving and Loading Models

```python
# Save your dimensionality reduction model
reducer.save_model('my_pca_model.pkl')

# Load the model later
from bioinformatics_dimensionalityreducer import BioinformaticsDimensionalityReducer
loaded_reducer = BioinformaticsDimensionalityReducer.load_model('my_pca_model.pkl')

# Continue using the loaded model
new_data_reduced = loaded_reducer.transform(new_data)
```

## Application Examples

### Gene Expression Analysis
```python
# Reduce dimensions of gene expression data
expression_reducer = BioinformaticsDimensionalityReducer(method='pca', n_components=2)
reduced_expression = expression_reducer.fit_transform(gene_expression)

# Plot with sample conditions
expression_reducer.plot_reduced_data(
    X_reduced=reduced_expression,
    y=sample_conditions,
    title='Gene Expression PCA'
)

# Identify genes with highest variance contribution
top_genes = expression_reducer.get_feature_loadings().iloc[0].abs().sort_values(ascending=False).head(10).index.tolist()
print(f"Top 10 genes contributing to PC1: {', '.join(top_genes)}")
```

### Single-Cell RNA-Seq Analysis
```python
# t-SNE is often used for scRNA-seq data
scrnaseq_reducer = BioinformaticsDimensionalityReducer(method='tsne', n_components=2)
reduced_cells = scrnaseq_reducer.fit_transform(
    single_cell_data,
    perplexity=30,
    n_iter=2000
)

# Plot with cell types
scrnaseq_reducer.plot_reduced_data(
    X_reduced=reduced_cells,
    y=cell_types,
    title='Single-Cell RNA-Seq t-SNE'
)
```

### Multi-Omics Data Integration
```python
# Combine PCA results from multiple omics datasets
pca_transcriptomics = BioinformaticsDimensionalityReducer(method='pca', n_components=10).fit_transform(transcriptomics_data)
pca_proteomics = BioinformaticsDimensionalityReducer(method='pca', n_components=10).fit_transform(proteomics_data)
pca_metabolomics = BioinformaticsDimensionalityReducer(method='pca', n_components=10).fit_transform(metabolomics_data)

# Combine the PCA results
import numpy as np
combined_pca = np.hstack([pca_transcriptomics, pca_proteomics, pca_metabolomics])

# Further reduce the combined data
final_reducer = BioinformaticsDimensionalityReducer(method='umap', n_components=2)
multi_omics_reduced = final_reducer.fit_transform(combined_pca)

# Plot the integrated result
final_reducer.plot_reduced_data(
    X_reduced=multi_omics_reduced,
    y=phenotypes,
    title='Multi-Omics Integration'
)
```

### Method Comparison
```python
# Compare different methods on the same dataset
methods = ['pca', 'tsne', 'umap', 'autoencoder']
results = {}

for method in methods:
    reducer = BioinformaticsDimensionalityReducer(method=method, n_components=2)
    results[method] = reducer.fit_transform(data)

# Plot comparison
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, method in enumerate(methods):
    reducer = BioinformaticsDimensionalityReducer(method=method)
    reducer.is_fitted = True  # Hack to allow plotting without refitting
    reducer.plot_reduced_data(
        X_reduced=results[method],
        y=sample_groups,
        title=f'{method.upper()}',
        ax=axes[i]
    )

plt.tight_layout()
plt.show()
```

## API Reference

### Main Class

#### `Bioinformatics Dimensionality Reducer`

**Parameters:**
- `method` (str): Dimensionality reduction method ('pca', 'lda', 'tsne', 'umap', 'autoencoder')
- `n_components` (int): Number of dimensions to reduce to
- `random_state` (int): Random seed for reproducibility
- `logger` (logging.Logger): Optional logger for tracking the reduction process

**Main Methods:**
- `fit(X, y=None, **kwargs)`: Fit the reducer to the data
- `transform(X)`: Transform data using the fitted reducer
- `fit_transform(X, y=None, **kwargs)`: Fit and transform in one step
- `plot_reduced_data(X=None, y=None, X_reduced=None, **kwargs)`: Plot the reduced data
- `plot_explained_variance(X=None, **kwargs)`: Plot variance explained (PCA only)
- `plot_feature_loadings(n_features=20, component_idx=0, **kwargs)`: Plot feature loadings
- `get_explained_variance_ratio()`: Get explained variance ratio (PCA only)
- `get_feature_loadings()`: Get feature loadings as DataFrame
- `save_model(filepath)`: Save reducer to file
- `load_model(filepath)`: Load reducer from file (class method)


## Acknowledgments

- This tool was inspired by the need for integrated dimensionality reduction approaches in bioinformatics
- Thanks to the scikit-learn, TensorFlow, and UMAP-learn teams for their excellent libraries
