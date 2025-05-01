import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Bioinformatics_DimensionalityReducer import BioinformaticsDimensionalityReducer

#Example #1
# 1. Load your bioinformatics data
# For example, gene expression data
gene_expression = pd.read_csv('gene_expression_data.csv', index_col=0)
metadata = pd.read_csv('sample_metadata.csv', index_col=0)

# Extract sample labels for coloring
sample_groups = metadata['condition']  # e.g., 'control', 'treatment'

# 2. Initialize and apply dimensionality reduction
# Using PCA
pca_reducer = BioinformaticsDimensionalityReducer(method='pca', n_components=2)
reduced_data_pca = pca_reducer.fit_transform(gene_expression)

# 3. Visualize the results
fig = pca_reducer.plot_reduced_data(X_reduced=reduced_data_pca, y=sample_groups, 
                                    title='Gene Expression PCA')
plt.savefig('gene_expression_pca.png')

# 4. Analyze the explained variance
fig_var = pca_reducer.plot_explained_variance()
plt.savefig('pca_explained_variance.png')

# 5. Look at feature loadings to identify important genes
fig_loadings = pca_reducer.plot_feature_loadings(n_features=15)
plt.savefig('pca_gene_loadings.png')

# 6. Try another method (t-SNE)
tsne_reducer = BioinformaticsDimensionalityReducer(method='tsne', n_components=2)
reduced_data_tsne = tsne_reducer.fit_transform(gene_expression)
fig_tsne = tsne_reducer.plot_reduced_data(X_reduced=reduced_data_tsne, y=sample_groups,
                                        title='Gene Expression t-SNE')
plt.savefig('gene_expression_tsne.png')

# 7. For supervised analysis, try LDA if you have class labels
if len(sample_groups.unique()) > 1:
    lda_reducer = BioinformaticsDimensionalityReducer(method='lda', n_components=2)
    reduced_data_lda = lda_reducer.fit_transform(gene_expression, sample_groups)
    fig_lda = lda_reducer.plot_reduced_data(X_reduced=reduced_data_lda, y=sample_groups,
                                          title='Gene Expression LDA')
    plt.savefig('gene_expression_lda.png')

# 8. Compare methods
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
pca_reducer.plot_reduced_data(X_reduced=reduced_data_pca, y=sample_groups, title='PCA', ax=axes[0])
tsne_reducer.plot_reduced_data(X_reduced=reduced_data_tsne, y=sample_groups, title='t-SNE', ax=axes[1])
lda_reducer.plot_reduced_data(X_reduced=reduced_data_lda, y=sample_groups, title='LDA', ax=axes[2])
plt.tight_layout()
plt.savefig('dimensionality_reduction_comparison.png')

# 9. Save models for later use
pca_reducer.save_model('pca_model.pkl')

#Example #2
#If process large dataset, use an autoencoder

# Using an autoencoder with a deeper architecture
autoencoder_reducer = BioinformaticsDimensionalityReducer(
    method='autoencoder', 
    n_components=2
)

# Define a more complex architecture with hidden layers
reduced_data_ae = autoencoder_reducer.fit_transform(
    gene_expression, 
    hidden_layers=[512, 256, 128],
    activation='relu',
    epochs=200,
    batch_size=32,
    learning_rate=0.001,
    dropout_rate=0.3
)

# Visualize the results
fig_ae = autoencoder_reducer.plot_reduced_data(
    X_reduced=reduced_data_ae, 
    y=sample_groups,
    title='Gene Expression Autoencoder'
)
plt.savefig('gene_expression_autoencoder.png')

