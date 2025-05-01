# Multi-Omics Integration and Feature selection Framework

A comprehensive Python framework for integrating multiple omics datasets, designed for flexibility and extensibility in multi-omics analysis.

## Overview

This framework provides tools for integrating different types of omics data (RNA-seq, metagenomics, epigenomics, variants) using various integration methods, with a special focus on deep learning approaches like Variational Autoencoders (VAEs).

## Features

- **Multiple omics data types support**:
  - RNA-seq
  - Metagenomics
  - Epigenomics (Metal-seq)
  - Variants

- **Comprehensive processing pipeline**:
  - Data loading and preprocessing
  - Batch effect correction
  - Feature selection
  - Data integration
  - Result visualization

- **Integration methods**:
  - Variational Autoencoders (VAEs)
  - Canonical Correlation Analysis (CCA)
  - Multi-Omics Factor Analysis (MOFA)

- **Flexible workflow system**:
  - Configuration-based workflow creation
  - Modular design for easy extension
  - Support for both same-modality and cross-modality integration


## Dependencies

- Python 3.8+
- NumPy
- Pandas
- PyTorch
- scikit-learn
- Matplotlib
- Seaborn
- UMAP

Additional dependencies for specific methods:
- pycombat (for ComBat batch correction)
- harmony-pytorch (for Harmony batch correction)
- mofapy2 (for MOFA integration)

## Quick Start

### Basic Usage

```python
from bioinformatics_featureselection import (
    DataLoader, WorkflowFactory, ResultVisualizer
)

# Load datasets
data_loader = DataLoader()
rnaseq = data_loader.load_dataset(
    data_path="rnaseq_data.csv",
    data_type="rnaseq",
    sample_metadata_path="sample_metadata.csv"
)
metagenomics = data_loader.load_dataset(
    data_path="metagenomics_data.csv",
    data_type="metagenomics",
    sample_metadata_path="sample_metadata.csv"
)

# Create workflow
workflow = WorkflowFactory.create_cross_modality_workflow([rnaseq, metagenomics])

# Execute workflow
integrated_data, info = workflow.execute()

# Visualize results
visualizer = ResultVisualizer()
labels = rnaseq.sample_metadata['condition'].values
pca_fig = visualizer.plot_pca(integrated_data, labels)
pca_fig.savefig("integrated_pca.png")
```

### Using Configuration Files

```python
import json
from bioinformatics_featureselection import IntegrationConfig, DataLoader

# Load configuration
with open("integration_config.json", 'r') as f:
    config = json.load(f)

# Create workflow from configuration
config_obj = IntegrationConfig(config)
workflow = config_obj.create_workflow()

# Load and add datasets
data_loader = DataLoader()
rnaseq = data_loader.load_dataset("rnaseq_data.csv", "rnaseq")
variants = data_loader.load_dataset("variants_data.csv", "variants")

workflow.add_dataset(rnaseq)
workflow.add_dataset(variants)

# Execute workflow
integrated_data, info = workflow.execute()
```

## Configuration Examples

### RNA-seq and Metagenomics Integration

```json
{
    "integration_type": "cross-modal",
    "preprocessing": [
        {
            "type": "feature_selection",
            "method": "variance",
            "n_features": 1000
        }
    ],
    "integration": {
        "method": "vae",
        "latent_dim": 20,
        "hidden_dims": [256, 128, 64],
        "epochs": 200
    }
}
```

### Multiple RNA-seq Datasets Integration

```json
{
    "integration_type": "same-modal",
    "preprocessing": [
        {
            "type": "batch_correction",
            "method": "combat",
            "batch_variable": "experiment"
        },
        {
            "type": "feature_selection",
            "method": "variance",
            "n_features": 2000
        }
    ],
    "integration": {
        "method": "vae",
        "latent_dim": 15,
        "hidden_dims": [128, 64]
    }
}
```

## Core Components

### Dataset Classes

- `OmicsDataset`: Base class for all omics data types
- `RNASeqDataset`: Specialized class for RNA-seq data
- `MetagenomicsDataset`: Specialized class for metagenomic data
- `EpigenomicsDataset`: Specialized class for epigenomics data (e.g., Metal-seq)
- `VariantsDataset`: Specialized class for variants data

### Integration Workflow

The typical integration workflow follows these steps:

1. **Data Preprocessing**: Each dataset type is preprocessed according to its specific requirements
2. **Batch Effect Correction**: For same-modality integration or addressing technical variation
3. **Pre-Integration Feature Selection**: Reducing dimensionality within each dataset before integration
4. **Data Integration**: Combining datasets using the selected integration method
5. **Post-Integration Feature Selection**: Identifying important features in the integrated space
6. **Analysis & Visualization**: Exploring and interpreting the integrated data

### Processing Components

- `BatchCorrector`: Base class for batch effect correction methods
  - `ComBatCorrector`: ComBat batch correction implementation
  - `HarmonyCorrector`: Harmony batch correction implementation

- `FeatureSelector`: Base class for feature selection methods
  - `VarianceSelector`: Select features based on variance
  - `DifferentialSelector`: Select features based on differential analysis

- `DataIntegrator`: Base class for integration methods
  - `CCAIntegrator`: Canonical Correlation Analysis integration
  - `MOFAIntegrator`: Multi-Omics Factor Analysis integration
  - `VAEIntegrator`: Variational Autoencoder integration

### Workflow System

- `IntegrationWorkflow`: Class for managing integration workflow
- `IntegrationConfig`: Configuration class for multi-omics integration
- `WorkflowFactory`: Factory class for creating common workflows

### Visualization Tools

- `ResultVisualizer`: Class for visualizing integration results
  - PCA visualization
  - UMAP visualization
  - Correlation heatmaps

## Advanced Usage

### Feature Selection Strategies

Feature selection can be performed at different stages of the integration process:

#### Pre-Integration Feature Selection

```python
from bioinformatics_featureselection import VarianceSelector, DifferentialSelector

# Select features before integration (per dataset)
variance_selector = VarianceSelector()
workflow.add_step(variance_selector)

# For differential analysis-based selection
diff_selector = DifferentialSelector()
diff_selector.group_variable = 'condition'  # Column in sample_metadata
workflow.add_step(diff_selector)
```

#### Post-Integration Feature Selection

```python
from bioinformatics_featureselection import IntegrationWorkflow, VAEIntegrator

# Create and execute workflow for integration
workflow = IntegrationWorkflow('cross-modal')
workflow.add_dataset(rnaseq)
workflow.add_dataset(metagenomics)

# Add integration step
integrator = VAEIntegrator(latent_dim=20)
workflow.add_step(integrator)

# Execute to get integrated data
integrated_data, info = workflow.execute()

# Post-integration feature selection
from sklearn.feature_selection import SelectKBest, f_classif

# Get sample labels
labels = rnaseq.sample_metadata['condition'].values

# Select features based on ANOVA F-value
selector = SelectKBest(f_classif, k=10)
selected_features = selector.fit_transform(integrated_data, labels)

# Get feature importance scores
feature_scores = pd.DataFrame({
    'Feature': [f'LD{i+1}' for i in range(integrated_data.shape[1])],
    'Score': selector.scores_
})
print("Top latent dimensions by importance:")
print(feature_scores.sort_values('Score', ascending=False).head(10))
```

### Using VAE Integration with Custom Architecture

```python
from bioinformatics_featureselection import VAEIntegrator

# Create VAE integrator with custom architecture
vae_integrator = VAEIntegrator(
    latent_dim=30,
    hidden_dims=[512, 256, 128, 64],
    epochs=300,
    batch_size=16,
    learning_rate=1e-4
)

# Add to workflow
workflow.add_step(vae_integrator)
```

### Analyzing Latent Space Correlations

```python
# Get the VAE model from integration results
vae_model = info['model']

# Get latent embeddings
with torch.no_grad():
    latent_mean, _ = vae_model.encode(combined_data)
    latent_dims = latent_mean.numpy()

# Create dataframe for latent dimensions
latent_df = pd.DataFrame(
    latent_dims,
    columns=[f"LD{i+1}" for i in range(latent_dims.shape[1])],
    index=dataset.data.index
)

# Compute correlation with original features
corr_matrix = pd.concat([dataset.data, latent_df], axis=1).corr()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Jiyang Jiang - jiyang.jiang@gmail.com

