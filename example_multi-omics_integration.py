"""
Configuration Examples for Multi-Omics Integration
-------------------------------------------------
This file contains various example configurations for different integration scenarios.
"""

import json

# Example 1: RNAseq and Metagenomics Integration (Cross-modal)
rnaseq_metagenomics_config = {
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
        "epochs": 200,
        "batch_size": 16,
        "learning_rate": 1e-3
    }
}

# Example 2: Multiple RNAseq Datasets Integration (Same-modal)
rnaseq_batch_integration_config = {
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
        "hidden_dims": [128, 64],
        "epochs": 150,
        "batch_size": 32,
        "learning_rate": 5e-4
    }
}

# Example 3: RNAseq and Metal-seq Integration (Same feature space)
rnaseq_metalseq_config = {
    "integration_type": "cross-modal",
    "preprocessing": [
        {
            "type": "feature_selection",
            "method": "differential",
            "group_variable": "condition",
            "n_features": 1500
        }
    ],
    "integration": {
        "method": "cca",  # Using CCA for datasets with similar feature space
    }
}

# Example 4: Multi-modal Integration (RNAseq, Variants, Metagenomics)
multimodal_config = {
    "integration_type": "cross-modal",
    "preprocessing": [
        {
            "type": "feature_selection",
            "method": "variance",
            "n_features": 800
        }
    ],
    "integration": {
        "method": "mofa",  # Using MOFA for multi-modal integration
    }
}

# Example 5: Advanced RNAseq and Variants Integration with batch correction
advanced_config = {
    "integration_type": "cross-modal",
    "preprocessing": [
        {
            "type": "batch_correction",
            "method": "harmony",
            "batch_variable": "batch"
        },
        {
            "type": "feature_selection",
            "method": "differential",
            "group_variable": "disease_status",
            "n_features": 1000
        }
    ],
    "integration": {
        "method": "vae",
        "latent_dim": 25,
        "hidden_dims": [512, 256, 128, 64],
        "epochs": 300,
        "batch_size": 16,
        "learning_rate": 1e-4
    }
}

# Function to save configurations to JSON files
def save_config_examples():
    """Save configuration examples to JSON files"""
    configs = {
        "rnaseq_metagenomics_config.json": rnaseq_metagenomics_config,
        "rnaseq_batch_integration_config.json": rnaseq_batch_integration_config,
        "rnaseq_metalseq_config.json": rnaseq_metalseq_config,
        "multimodal_config.json": multimodal_config,
        "advanced_config.json": advanced_config
    }
    
    for filename, config in configs.items():
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved {filename}")

# Example usage of a configuration
def use_config_example():
    """Example of using a configuration file with the framework"""
    from multi_omics_integration import (
        DataLoader, WorkflowFactory, ResultVisualizer
    )
    
    # Load configuration
    with open("rnaseq_metagenomics_config.json", 'r') as f:
        config = json.load(f)
    
    # Load datasets
    data_loader = DataLoader()
    
    # Load RNA-seq dataset
    rnaseq = data_loader.load_dataset(
        data_path="rnaseq_data.csv",
        data_type="rnaseq",
        sample_metadata_path="sample_metadata.csv"
    )
    
    # Load metagenomics dataset
    metagenomics = data_loader.load_dataset(
        data_path="metagenomics_data.csv",
        data_type="metagenomics",
        sample_metadata_path="sample_metadata.csv"
    )
    
    # Create integration workflow from configuration
    workflow_factory = WorkflowFactory()
    workflow = workflow_factory.create_workflow_from_config("rnaseq_metagenomics_config.json")
    
    # Add datasets to workflow
    workflow.add_dataset(rnaseq)
    workflow.add_dataset(metagenomics)
    
    # Execute workflow
    integrated_data, info = workflow.execute()
    
    # Visualize results
    visualizer = ResultVisualizer()
    labels = rnaseq.sample_metadata['condition'].values
    
    # Plot and save PCA
    pca_fig = visualizer.plot_pca(integrated_data, labels, 
                                 "PCA of Integrated RNA-seq and Metagenomics")
    pca_fig.savefig("integrated_pca.png")
    
    print("Integration complete!")


if __name__ == "__main__":
    save_config_examples()
    # use_config_example()  # Uncomment to run the example

