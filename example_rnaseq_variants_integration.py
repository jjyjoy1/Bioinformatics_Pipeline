"""
Example: Integration of RNAseq and Variants Data
-----------------------------------------------
This example demonstrates how to use the Multi-Omics Integration Framework
to integrate RNAseq and Variants data using a VAE approach.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Import our framework
# In a real scenario, you would import from the package
# from multi_omics_integration import *

# For this example, assuming the framework is in the same file
# Execute this in the same environment as the framework

def integrate_rnaseq_and_variants():
    """Example function to integrate RNAseq and variants data"""
    print("Starting integration of RNAseq and Variants data...")
    
    # Step 1: Load and prepare data
    # In a real scenario, you would load from files
    # Here we create sample data for demonstration
    
    # Sample RNAseq data (20 samples, 1000 genes)
    rnaseq_data = pd.DataFrame(
        np.random.negative_binomial(10, 0.5, size=(20, 1000)),
        columns=[f"gene_{i}" for i in range(1000)],
        index=[f"sample_{i}" for i in range(20)]
    )
    
    # Sample variants data (20 samples, 500 variants)
    variants_data = pd.DataFrame(
        np.random.randint(0, 3, size=(20, 500)),  # 0, 1, 2 for genotypes
        columns=[f"variant_{i}" for i in range(500)],
        index=[f"sample_{i}" for i in range(20)]
    )
    
    # Sample metadata
    sample_metadata = pd.DataFrame({
        'condition': np.random.choice(['control', 'case'], size=20),
        'batch': np.random.choice(['batch1', 'batch2', 'batch3'], size=20),
        'gender': np.random.choice(['M', 'F'], size=20),
        'age': np.random.randint(20, 80, size=20)
    }, index=[f"sample_{i}" for i in range(20)])
    
    # Step 2: Create dataset objects
    rnaseq_dataset = RNASeqDataset(rnaseq_data, None, sample_metadata)
    variants_dataset = VariantsDataset(variants_data, None, sample_metadata)
    
    # Step 3: Preprocess datasets
    print("Preprocessing datasets...")
    
    # Preprocess RNAseq data
    rnaseq_dataset.preprocess()  # Filter low-expression genes
    rnaseq_dataset.normalize('log')  # Log-normalize
    
    # Preprocess variants data
    variants_dataset.preprocess(min_maf=0.05)  # Filter by minor allele frequency
    
    # Step 4: Configure the integration workflow
    print("Configuring integration workflow...")
    
    # Create a config dictionary
    config = {
        "integration_type": "cross-modal",
        "preprocessing": [
            {
                "type": "feature_selection",
                "method": "variance",
                "n_features": 500  # Select top 500 features by variance
            }
        ],
        "integration": {
            "method": "vae",
            "latent_dim": 15,  # 15-dimensional latent space
            "hidden_dims": [128, 64],
            "epochs": 50,  # Reduced for demonstration
            "batch_size": 5,
            "learning_rate": 1e-3
        }
    }
    
    # Create integration config
    integration_config = IntegrationConfig(config)
    
    # Create workflow
    workflow = integration_config.create_workflow()
    
    # Add datasets
    workflow.add_dataset(rnaseq_dataset)
    workflow.add_dataset(variants_dataset)
    
    # Step 5: Execute workflow
    print("Executing integration workflow...")
    integrated_data, info = workflow.execute()
    
    # Step 6: Analyze and visualize results
    print("Analyzing and visualizing results...")
    
    # Get sample condition for coloring
    labels = sample_metadata['condition'].values
    
    # Visualize
    visualizer = ResultVisualizer()
    
    # PCA plot
    pca_fig = visualizer.plot_pca(integrated_data, labels, 
                                 "PCA of Integrated RNA-seq and Variants")
    
    # UMAP plot
    umap_fig = visualizer.plot_umap(integrated_data, labels,
                                   "UMAP of Integrated RNA-seq and Variants")
    
    # Display figures
    pca_fig.show()
    umap_fig.show()
    
    # Step 7: Examine correlations between original features and latent space
    print("Examining correlations with latent space...")
    
    # Get the VAE model
    vae_model = info['model']
    
    # Get original features
    rnaseq_features = rnaseq_dataset.data.columns.tolist()
    variant_features = variants_dataset.data.columns.tolist()
    
    # Compute correlations between original features and latent dimensions
    with torch.no_grad():
        # Get latent embeddings
        rnaseq_tensor = torch.tensor(rnaseq_dataset.data.values, dtype=torch.float32)
        variants_tensor = torch.tensor(variants_dataset.data.values, dtype=torch.float32)
        
        # Combine data
        combined_data = torch.cat([rnaseq_tensor, variants_tensor], dim=1)
        
        # Get latent embeddings
        latent_mean, _ = vae_model.encode(combined_data)
        latent_dims = latent_mean.numpy()
    
    # Create dataframe for latent dimensions
    latent_df = pd.DataFrame(
        latent_dims,
        columns=[f"LD{i+1}" for i in range(latent_dims.shape[1])],
        index=rnaseq_dataset.data.index
    )
    
    # Combine original data with latent dimensions
    combined_df = pd.concat([rnaseq_dataset.data, variants_dataset.data, latent_df], axis=1)
    
    # Compute correlation matrix
    corr_matrix = combined_df.corr()
    
    # Extract correlations between original features and latent dimensions
    latent_cols = [f"LD{i+1}" for i in range(latent_dims.shape[1])]
    feature_latent_corr = corr_matrix.loc[rnaseq_features + variant_features, latent_cols]
    
    # Find top correlated features for each latent dimension
    for ld in latent_cols:
        top_features = feature_latent_corr[ld].abs().nlargest(5)
        print(f"\nTop features correlated with {ld}:")
        for feature, corr in top_features.items():
            feature_type = "RNAseq" if feature in rnaseq_features else "Variant"
            print(f"  {feature} ({feature_type}): {corr:.3f}")
    
    print("\nIntegration analysis complete!")
    return integrated_data, info, workflow


if __name__ == "__main__":
    # Run the example
    integrated_data, info, workflow = integrate_rnaseq_and_variants()

