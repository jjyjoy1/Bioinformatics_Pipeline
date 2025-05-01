"""
Multi-Omics Integration Framework
--------------------------------
A comprehensive framework for integrating multi-omics data, including RNA-seq, 
metagenomics, epigenomics (Metal-seq), and variants datasets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from 
# ------------------------------------------------------------------------
# Base Classes
# ------------------------------------------------------------------------

class OmicsDataset(ABC):
    """Base class for all omics data types"""
    
    def __init__(self, data_matrix: pd.DataFrame, 
                 feature_metadata: Optional[pd.DataFrame] = None, 
                 sample_metadata: Optional[pd.DataFrame] = None):
        """
        Initialize OmicsDataset
        
        Parameters:
        -----------
        data_matrix : pd.DataFrame
            Data matrix with samples as rows and features as columns
        feature_metadata : pd.DataFrame, optional
            Metadata for features
        sample_metadata : pd.DataFrame, optional
            Metadata for samples
        """
        self.data = data_matrix
        self.feature_metadata = feature_metadata
        self.sample_metadata = sample_metadata
        self.name = "generic"
        
    def filter_features(self, min_prevalence: float = 0.1, min_abundance: float = 0.0):
        """Filter features based on prevalence and abundance"""
        # Calculate prevalence (fraction of samples where feature is present)
        prevalence = (self.data > 0).mean(axis=0)
        # Calculate mean abundance across samples
        abundance = self.data.mean(axis=0)
        
        # Apply filters
        keep = (prevalence >= min_prevalence) & (abundance >= min_abundance)
        self.data = self.data.loc[:, keep]
        
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata.loc[keep, :]
            
        print(f"Filtered features: {sum(~keep)} removed, {sum(keep)} remaining")
        return self
    
    def normalize(self, method: str = 'log'):
        """Normalize data using specified method"""
        if method == 'log':
            # Add small constant to avoid log(0)
            self.data = np.log1p(self.data)
        elif method == 'scale':
            # Scale each feature to zero mean and unit variance
            scaler = StandardScaler()
            self.data = pd.DataFrame(
                scaler.fit_transform(self.data),
                index=self.data.index,
                columns=self.data.columns
            )
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return self
    
    def subset_samples(self, sample_indices):
        """Subset dataset to specific samples"""
        self.data = self.data.iloc[sample_indices, :]
        if self.sample_metadata is not None:
            self.sample_metadata = self.sample_metadata.iloc[sample_indices, :]
        return self
    
    def subset_features(self, feature_indices):
        """Subset dataset to specific features"""
        self.data = self.data.iloc[:, feature_indices]
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata.iloc[feature_indices, :]
        return self

    @abstractmethod
    def preprocess(self):
        """Preprocess data (to be implemented by subclasses)"""
        pass
    
    def __len__(self):
        """Return number of samples"""
        return self.data.shape[0]
    
    def feature_count(self):
        """Return number of features"""
        return self.data.shape[1]


class RNASeqDataset(OmicsDataset):
    """Class for RNA-seq data"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "rnaseq"
    
    def preprocess(self, min_counts: int = 10, min_samples: int = 3):
        """Preprocess RNA-seq data"""
        # Filter out low-expression genes
        keep = (self.data >= min_counts).sum(axis=0) >= min_samples
        self.data = self.data.loc[:, keep]
        
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata.loc[keep, :]
            
        print(f"Filtered genes: {sum(~keep)} removed, {sum(keep)} remaining")
        return self


class MetagenomicsDataset(OmicsDataset):
    """Class for metagenomic data"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "metagenomics"
    
    def preprocess(self, min_prevalence: float = 0.1):
        """Preprocess metagenomic data"""
        # Filter by prevalence
        prevalence = (self.data > 0).mean(axis=0)
        keep = prevalence >= min_prevalence
        
        self.data = self.data.loc[:, keep]
        
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata.loc[keep, :]
            
        print(f"Filtered taxa: {sum(~keep)} removed, {sum(keep)} remaining")
        return self


class EpigenomicsDataset(OmicsDataset):
    """Class for epigenomics data (e.g., Metal-seq)"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "epigenomics"
    
    def preprocess(self, min_variance: float = 0.1):
        """Preprocess epigenomics data"""
        # Filter features with low variance
        variances = self.data.var(axis=0)
        keep = variances >= min_variance
        
        self.data = self.data.loc[:, keep]
        
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata.loc[keep, :]
            
        print(f"Filtered features: {sum(~keep)} removed, {sum(keep)} remaining")
        return self


class VariantsDataset(OmicsDataset):
    """Class for genetic variants data"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "variants"
    
    def preprocess(self, min_maf: float = 0.05):
        """Preprocess variants data"""
        # Filter by minor allele frequency
        maf = self.data.mean(axis=0)
        maf = np.minimum(maf, 1 - maf)  # Get minor allele frequency
        keep = maf >= min_maf
        
        self.data = self.data.loc[:, keep]
        
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata.loc[keep, :]
            
        print(f"Filtered variants: {sum(~keep)} removed, {sum(keep)} remaining")
        return self


# ------------------------------------------------------------------------
# Data Loading and Management
# ------------------------------------------------------------------------

class DataLoader:
    """Factory class for loading different omics datasets"""
    
    @staticmethod
    def load_dataset(data_path: str, 
                     data_type: str, 
                     feature_metadata_path: Optional[str] = None,
                     sample_metadata_path: Optional[str] = None) -> OmicsDataset:
        """
        Load dataset from file
        
        Parameters:
        -----------
        data_path : str
            Path to data matrix file
        data_type : str
            Type of data ('rnaseq', 'metagenomics', 'epigenomics', 'variants')
        feature_metadata_path : str, optional
            Path to feature metadata file
        sample_metadata_path : str, optional
            Path to sample metadata file
            
        Returns:
        --------
        OmicsDataset
            Appropriate OmicsDataset subclass instance
        """
        # Load data matrix
        data_matrix = pd.read_csv(data_path, index_col=0)
        
        # Load feature metadata if available
        feature_metadata = None
        if feature_metadata_path:
            feature_metadata = pd.read_csv(feature_metadata_path, index_col=0)
            
        # Load sample metadata if available
        sample_metadata = None
        if sample_metadata_path:
            sample_metadata = pd.read_csv(sample_metadata_path, index_col=0)
        
        # Create appropriate dataset class
        if data_type.lower() == 'rnaseq':
            return RNASeqDataset(data_matrix, feature_metadata, sample_metadata)
        elif data_type.lower() == 'metagenomics':
            return MetagenomicsDataset(data_matrix, feature_metadata, sample_metadata)
        elif data_type.lower() == 'epigenomics':
            return EpigenomicsDataset(data_matrix, feature_metadata, sample_metadata)
        elif data_type.lower() == 'variants':
            return VariantsDataset(data_matrix, feature_metadata, sample_metadata)
        else:
            raise ValueError(f"Unknown data type: {data_type}")


# ------------------------------------------------------------------------
# Batch Effect Correction
# ------------------------------------------------------------------------

class BatchCorrector(ABC):
    """Base class for batch effect correction methods"""
    
    @abstractmethod
    def correct(self, dataset: OmicsDataset, batch_variable: str) -> OmicsDataset:
        """
        Correct batch effects
        
        Parameters:
        -----------
        dataset : OmicsDataset
            Dataset to correct
        batch_variable : str
            Column name in sample_metadata that contains batch information
            
        Returns:
        --------
        OmicsDataset
            Batch-corrected dataset
        """
        pass


class ComBatCorrector(BatchCorrector):
    """ComBat batch correction"""
    
    def correct(self, dataset: OmicsDataset, batch_variable: str) -> OmicsDataset:
        """Correct batch effects using ComBat"""
        from combat.pycombat import pycombat
        
        if dataset.sample_metadata is None:
            raise ValueError("Sample metadata required for batch correction")
        
        if batch_variable not in dataset.sample_metadata.columns:
            raise ValueError(f"Batch variable {batch_variable} not found in sample metadata")
        
        # Get batch information
        batch = dataset.sample_metadata[batch_variable].values
        
        # Apply ComBat
        corrected_data = pycombat(dataset.data.values.T, batch).T
        
        # Create new dataframe with corrected values
        corrected_df = pd.DataFrame(
            corrected_data,
            index=dataset.data.index,
            columns=dataset.data.columns
        )
        
        # Create new dataset with corrected data
        corrected_dataset = type(dataset)(
            corrected_df,
            dataset.feature_metadata,
            dataset.sample_metadata
        )
        
        return corrected_dataset


class HarmonyCorrector(BatchCorrector):
    """Harmony batch correction"""
    
    def correct(self, dataset: OmicsDataset, batch_variable: str) -> OmicsDataset:
        """Correct batch effects using Harmony"""
        try:
            from harmony import harmonize
        except ImportError:
            raise ImportError("harmony-pytorch package required for Harmony batch correction")
        
        if dataset.sample_metadata is None:
            raise ValueError("Sample metadata required for batch correction")
        
        if batch_variable not in dataset.sample_metadata.columns:
            raise ValueError(f"Batch variable {batch_variable} not found in sample metadata")
        
        # First reduce dimensions with PCA
        pca = PCA(n_components=20)
        pca_data = pca.fit_transform(dataset.data)
        
        # Apply Harmony
        batch = dataset.sample_metadata[batch_variable].values
        harmony_embeddings = harmonize(pca_data, batch, max_iter_harmony=10)
        
        # Project back to original space
        corrected_data = np.dot(harmony_embeddings, pca.components_) + pca.mean_
        
        # Create new dataframe with corrected values
        corrected_df = pd.DataFrame(
            corrected_data,
            index=dataset.data.index,
            columns=dataset.data.columns
        )
        
        # Create new dataset with corrected data
        corrected_dataset = type(dataset)(
            corrected_df,
            dataset.feature_metadata,
            dataset.sample_metadata
        )
        
        return corrected_dataset


# ------------------------------------------------------------------------
# Feature Selection
# ------------------------------------------------------------------------

class FeatureSelector(ABC):
    """Base class for feature selection methods"""
    
    @abstractmethod
    def select(self, dataset: OmicsDataset, n_features: int = 1000) -> List[str]:
        """
        Select features
        
        Parameters:
        -----------
        dataset : OmicsDataset
            Dataset for feature selection
        n_features : int
            Number of features to select
            
        Returns:
        --------
        List[str]
            List of selected feature names
        """
        pass


class VarianceSelector(FeatureSelector):
    """Select features based on variance"""
    
    def select(self, dataset: OmicsDataset, n_features: int = 1000) -> List[str]:
        """Select features with highest variance"""
        variances = dataset.data.var(axis=0)
        selected = variances.nlargest(n_features).index.tolist()
        return selected


class DifferentialSelector(FeatureSelector):
    """Select features based on differential analysis"""
    
    def select(self, dataset: OmicsDataset, group_variable: str, n_features: int = 1000) -> List[str]:
        """Select features based on differential analysis between groups"""
        if dataset.sample_metadata is None:
            raise ValueError("Sample metadata required for differential selection")
            
        if group_variable not in dataset.sample_metadata.columns:
            raise ValueError(f"Group variable {group_variable} not found in sample metadata")
        
        from scipy.stats import ttest_ind
        
        # Get group information
        groups = dataset.sample_metadata[group_variable].unique()
        if len(groups) < 2:
            raise ValueError("At least two groups required for differential selection")
        
        # Perform t-test for each feature
        p_values = []
        for col in dataset.data.columns:
            group1 = dataset.data.loc[dataset.sample_metadata[group_variable] == groups[0], col]
            group2 = dataset.data.loc[dataset.sample_metadata[group_variable] == groups[1], col]
            _, p_value = ttest_ind(group1, group2, nan_policy='omit')
            p_values.append(p_value)
        
        # Convert to Series for easier handling
        p_value_series = pd.Series(p_values, index=dataset.data.columns)
        
        # Select features with lowest p-values
        selected = p_value_series.nsmallest(n_features).index.tolist()
        return selected


# ------------------------------------------------------------------------
# Multi-Omics Integration
# ------------------------------------------------------------------------

class DataIntegrator(ABC):
    """Base class for multi-omics data integration methods"""
    
    @abstractmethod
    def integrate(self, datasets: List[OmicsDataset]) -> Tuple[np.ndarray, Dict]:
        """
        Integrate multiple omics datasets
        
        Parameters:
        -----------
        datasets : List[OmicsDataset]
            List of datasets to integrate
            
        Returns:
        --------
        Tuple[np.ndarray, Dict]
            Integrated data and additional information
        """
        pass


class CCAIntegrator(DataIntegrator):
    """Canonical Correlation Analysis integration"""
    
    def integrate(self, datasets: List[OmicsDataset]) -> Tuple[np.ndarray, Dict]:
        """Integrate datasets using CCA"""
        from sklearn.cross_decomposition import CCA
        
        if len(datasets) != 2:
            raise ValueError("CCA integration requires exactly 2 datasets")
        
        # Extract data matrices
        X = datasets[0].data.values
        Y = datasets[1].data.values
        
        # Apply CCA
        n_components = min(X.shape[1], Y.shape[1], X.shape[0], Y.shape[0])
        n_components = min(n_components, 10)  # Limit to 10 components
        
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(X, Y)
        
        # Concatenate CCA components
        integrated_data = np.hstack([X_c, Y_c])
        
        # Return integrated data and CCA model
        return integrated_data, {'model': cca}


class MOFAIntegrator(DataIntegrator):
    """Multi-Omics Factor Analysis integration"""
    
    def integrate(self, datasets: List[OmicsDataset]) -> Tuple[np.ndarray, Dict]:
        """Integrate datasets using MOFA"""
        try:
            import mofapy2.create_mofa as cm
            import mofapy2.run_mofa as rm
        except ImportError:
            raise ImportError("mofapy2 package required for MOFA integration")
        
        # Create MOFA object
        mofa = cm.create_mofa()
        
        # Add data views
        for i, dataset in enumerate(datasets):
            mofa.add_view(dataset.data.values, view_name=dataset.name)
        
        # Set options
        mofa.set_train_options(
            iter=1000,
            convergence_mode="fast",
            dropR2=0.001,
            verbose=False
        )
        
        # Set model options
        mofa.set_model_options(
            factors=10,
            spikeslab_factors=True
        )
        
        # Run MOFA
        mofa_model = rm.run_mofa(mofa)
        
        # Get factors
        Z = mofa_model.get_factors()
        
        # Return integrated data (factors) and MOFA model
        return Z, {'model': mofa_model}


class VAEIntegrator(DataIntegrator):
    """Variational Autoencoder integration"""
    
    def __init__(self, 
                 latent_dim: int = 10, 
                 hidden_dims: List[int] = [128, 64],
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3):
        """
        Initialize VAE integrator
        
        Parameters:
        -----------
        latent_dim : int
            Dimension of latent space
        hidden_dims : List[int]
            Dimensions of hidden layers
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def integrate(self, datasets: List[OmicsDataset]) -> Tuple[np.ndarray, Dict]:
        """Integrate datasets using VAE"""
        # Concatenate all datasets
        data_list = [dataset.data.values for dataset in datasets]
        
        # Get feature dimensions for each dataset
        feature_dims = [data.shape[1] for data in data_list]
        total_features = sum(feature_dims)
        
        # Concatenate data horizontally
        combined_data = np.hstack(data_list)
        
        # Create PyTorch dataset and dataloader
        tensor_data = torch.tensor(combined_data, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create and train VAE model
        vae = MultiOmicsVAE(input_dim=total_features, 
                           feature_dims=feature_dims,
                           latent_dim=self.latent_dim,
                           hidden_dims=self.hidden_dims)
        
        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)
        
        # Training loop
        vae.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch[0]
                
                # Forward pass
                recon_x, mu, log_var = vae(x)
                
                # Compute loss
                recon_loss = F.mse_loss(recon_x, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + kl_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss / len(dataloader.dataset):.4f}")
        
        # Generate latent embeddings for all samples
        vae.eval()
        with torch.no_grad():
            latent_embeddings = vae.encode(torch.tensor(combined_data, dtype=torch.float32))[0].numpy()
        
        return latent_embeddings, {'model': vae, 'feature_dims': feature_dims}


class MultiOmicsVAE(nn.Module):
    """VAE model for multi-omics integration"""
    
    def __init__(self, 
                 input_dim: int, 
                 feature_dims: List[int],
                 latent_dim: int = 10, 
                 hidden_dims: List[int] = [128, 64]):
        """
        Initialize VAE model
        
        Parameters:
        -----------
        input_dim : int
            Total dimension of input (sum of all omics)
        feature_dims : List[int]
            Dimensions of each omics dataset
        latent_dim : int
            Dimension of latent space
        hidden_dims : List[int]
            Dimensions of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_dims = feature_dims
        
        # Build encoder
        modules = []
        
        # Input layer
        input_size = input_dim
        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_size, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.LeakyReLU())
            input_size = h_dim
            
        self.encoder = nn.Sequential(*modules)
        
        # Latent space
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder
        modules = []
        
        # Reverse hidden dimensions
        hidden_dims = hidden_dims[::-1]
        
        # From latent to first hidden
        modules.append(nn.Linear(latent_dim, hidden_dims[0]))
        modules.append(nn.BatchNorm1d(hidden_dims[0]))
        modules.append(nn.LeakyReLU())
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            modules.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            modules.append(nn.BatchNorm1d(hidden_dims[i + 1]))
            modules.append(nn.LeakyReLU())
            
        self.decoder_hidden = nn.Sequential(*modules)
        
        # Output layers - one for each omics type
        self.output_layers = nn.ModuleList([
            nn.Linear(hidden_dims[-1], dim) for dim in feature_dims
        ])
        
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """Reparameterization trick"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent variable to reconstructed input"""
        h = self.decoder_hidden(z)
        
        # Get reconstruction for each omics type
        recons = []
        start_idx = 0
        
        for i, layer in enumerate(self.output_layers):
            recon_i = layer(h)
            recons.append(recon_i)
            
        # Concatenate all reconstructions
        recon_x = torch.cat(recons, dim=1)
        return recon_x
    
    def forward(self, x):
        """Forward pass"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var


# ------------------------------------------------------------------------
# Visualization and Analysis
# ------------------------------------------------------------------------

class ResultVisualizer:
    """Class for visualizing integrated results"""
    
    @staticmethod
    def plot_pca(data: np.ndarray, 
                 labels: Optional[np.ndarray] = None, 
                 title: str = "PCA of Integrated Data"):
        """Plot PCA of integrated data"""
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(data)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Plot with color labels
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=label)
            plt.legend()
        else:
            # Plot without labels
            plt.scatter(pca_result[:, 0], pca_result[:, 1])
            
        plt.title(title)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.grid(alpha=0.3)
        
        return plt.gcf()
    
    @staticmethod
    def plot_umap(data: np.ndarray, 
                  labels: Optional[np.ndarray] = None,
                  title: str = "UMAP of Integrated Data"):
        """Plot UMAP of integrated data"""
        # Apply UMAP
        reducer = umap.UMAP()
        umap_result = reducer.fit_transform(data)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            # Plot with color labels
            for label in np.unique(labels):
                mask = labels == label
                plt.scatter(umap_result[mask, 0], umap_result[mask, 1], label=label)
            plt.legend()
        else:
            # Plot without labels
            plt.scatter(umap_result[:, 0], umap_result[:, 1])
            
        plt.title(title)
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.grid(alpha=0.3)
        
        return plt.gcf()
    
    @staticmethod
    def plot_correlation_heatmap(datasets: List[OmicsDataset], 
                                title: str = "Feature Correlation Heatmap"):
        """Plot correlation heatmap between datasets"""
        # Get feature correlations
        correlations = []
        
        # For each pair of datasets
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                # Get common samples
                common_samples = datasets[i].data.index.intersection(datasets[j].data.index)
                
                # Check if there are enough common samples
                if len(common_samples) < 5:
                    print(f"Warning: Not enough common samples between datasets {i} and {j}")
                    continue
                
                # Get data for common samples
                data_i = datasets[i].data.loc[common_samples]
                data_j = datasets[j].data.loc[common_samples]
                
                # Calculate correlation matrix
                corr_matrix = data_i.T.corr(data_j.T)
                correlations.append((i, j, corr_matrix))
        
        # Plot heatmaps for each pair
        n_pairs = len(correlations)
        if n_pairs == 0:
            print("No correlations to plot")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, n_pairs, figsize=(7*n_pairs, 6))
        if n_pairs == 1:
            axes = [axes]
            
        for idx, (i, j, corr) in enumerate(correlations):
            # Plot heatmap
            sns.heatmap(corr, cmap="coolwarm", center=0, 
                       vmin=-1, vmax=1, ax=axes[idx])
            axes[idx].set_title(f"{datasets[i].name} vs {datasets[j].name}")
            
        plt.tight_layout()
        return fig


# ------------------------------------------------------------------------
# Multi-Omics Integration Workflow
# ------------------------------------------------------------------------

class IntegrationWorkflow:
    """Class for managing multi-omics integration workflow"""
    
    def __init__(self, integration_type: str = 'cross-modal'):
        """
        Initialize workflow
        
        Parameters:
        -----------
        integration_type : str
            Type of integration ('cross-modal' or 'same-modal')
        """
        self.integration_type = integration_type
        self.steps = []
        self.datasets = []
        
    def add_dataset(self, dataset: OmicsDataset):
        """Add dataset to workflow"""
        self.datasets.append(dataset)
        return self
        
    def add_step(self, step: Any):
        """Add processing step to workflow"""
        self.steps.append(step)
        return self
    
    def execute(self):
        """Execute workflow"""
        result = self.datasets
        for step in self.steps:
            if isinstance(step, BatchCorrector):
                # Apply batch correction to each dataset
                corrected_datasets = []
                for dataset in result:
                    if hasattr(dataset, 'sample_metadata') and dataset.sample_metadata is not None:
                        batch_variable = step.batch_variable if hasattr(step, 'batch_variable') else 'batch'
                        if batch_variable in dataset.sample_metadata.columns:
                            corrected_dataset = step.correct(dataset, batch_variable)
                            corrected_datasets.append(corrected_dataset)
                        else:
                            # Skip correction if batch variable not found
                            print(f"Batch variable {batch_variable} not found in {dataset.name}, skipping correction")
                            corrected_datasets.append(dataset)
                    else:
                        # Skip correction if no sample metadata
                        print(f"No sample metadata in {dataset.name}, skipping correction")
                        corrected_datasets.append(dataset)
                result = corrected_datasets
            elif isinstance(step, FeatureSelector):
                # Apply feature selection to each dataset
                selected_datasets = []
                for dataset in result:
                    # Select features
                    selected_features = step.select(dataset)
                    # Subset dataset to selected features
                    selected_dataset = dataset.subset_features([dataset.data.columns.get_loc(f) for f in selected_features])
                    selected_datasets.append(selected_dataset)
                result = selected_datasets
            elif isinstance(step, DataIntegrator):
                # Integrate datasets
                integrated_data, info = step.integrate(result)
                # Store integration result
                self.integrated_data = integrated_data
                self.integration_info = info
                return integrated_data, info
            else:
                # Apply custom step
                result = step(result)
        
        return result


# ------------------------------------------------------------------------
# Configuration and Factory Classes
# ------------------------------------------------------------------------

class IntegrationConfig:
    """Configuration class for multi-omics integration"""
    
    def __init__(self, config_dict: Dict = None):
        """Initialize from config dictionary"""
        self.config = config_dict or {}
        
    def create_workflow(self) -> IntegrationWorkflow:
        """Create workflow from configuration"""
        # Get integration type
        integration_type = self.config.get('integration_type', 'cross-modal')
        
        # Create workflow
        workflow = IntegrationWorkflow(integration_type)
        
        # Add preprocessing steps
        if 'preprocessing' in self.config:
            for step_config in self.config['preprocessing']:
                step_type = step_config.get('type')
                
                if step_type == 'batch_correction':
                    method = step_config.get('method', 'combat')
                    batch_variable = step_config.get('batch_variable', 'batch')
                    
                    if method == 'combat':
                        corrector = ComBatCorrector()
                    elif method == 'harmony':
                        corrector = HarmonyCorrector()
                    else:
                        raise ValueError(f"Unknown batch correction method: {method}")
                    
                    # Add batch variable
                    corrector.batch_variable = batch_variable
                    workflow.add_step(corrector)
                    
                elif step_type == 'feature_selection':
                    method = step_config.get('method', 'variance')
                    n_features = step_config.get('n_features', 1000)
                    
                    if method == 'variance':
                        selector = VarianceSelector()
                    elif method == 'differential':
                        selector = DifferentialSelector()
                        # Add group variable
                        selector.group_variable = step_config.get('group_variable', 'group')
                    else:
                        raise ValueError(f"Unknown feature selection method: {method}")
                    
                    workflow.add_step(selector)
        
        # Add integration step
        if 'integration' in self.config:
            integration_config = self.config['integration']
            method = integration_config.get('method', 'vae')
            
            if method == 'cca':
                integrator = CCAIntegrator()
            elif method == 'mofa':
                integrator = MOFAIntegrator()
            elif method == 'vae':
                # Get VAE parameters
                latent_dim = integration_config.get('latent_dim', 10)
                hidden_dims = integration_config.get('hidden_dims', [128, 64])
                epochs = integration_config.get('epochs', 100)
                batch_size = integration_config.get('batch_size', 32)
                learning_rate = integration_config.get('learning_rate', 1e-3)
                
                integrator = VAEIntegrator(
                    latent_dim=latent_dim,
                    hidden_dims=hidden_dims,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
            else:
                raise ValueError(f"Unknown integration method: {method}")
                
            workflow.add_step(integrator)
            
        return workflow


class WorkflowFactory:
    """Factory class for creating workflows"""
    
    @staticmethod
    def create_same_modality_workflow(datasets: List[OmicsDataset]) -> IntegrationWorkflow:
        """Create workflow for same modality integration"""
        workflow = IntegrationWorkflow('same-modal')
        
        # Add datasets
        for dataset in datasets:
            workflow.add_dataset(dataset)
        
        # Add batch correction step
        corrector = ComBatCorrector()
        corrector.batch_variable = 'experiment'
        workflow.add_step(corrector)
        
        # Add feature selection step
        selector = VarianceSelector()
        workflow.add_step(selector)
        
        # Add integration step
        integrator = VAEIntegrator(latent_dim=20, hidden_dims=[128, 64], epochs=200)
        workflow.add_step(integrator)
        
        return workflow
    
    @staticmethod
    def create_cross_modality_workflow(datasets: List[OmicsDataset]) -> IntegrationWorkflow:
        """Create workflow for cross modality integration"""
        workflow = IntegrationWorkflow('cross-modal')
        
        # Add datasets
        for dataset in datasets:
            workflow.add_dataset(dataset)
        
        # Add feature selection step for each dataset
        selector = VarianceSelector()
        workflow.add_step(selector)
        
        # Add integration step
        integrator = VAEIntegrator(latent_dim=30, hidden_dims=[256, 128, 64], epochs=300)
        workflow.add_step(integrator)
        
        return workflow
    
    @staticmethod
    def create_workflow_from_config(config_file: str) -> IntegrationWorkflow:
        """Create workflow from configuration file"""
        import json
        
        # Load configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Create config object
        config_obj = IntegrationConfig(config)
        
        # Create workflow
        workflow = config_obj.create_workflow()
        
        return workflow


# ------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------

def example_usage():
    """Example usage of the multi-omics integration framework"""
    # Load datasets
    data_loader = DataLoader()
    
    # Load RNA-seq dataset
    rnaseq = data_loader.load_dataset(
        data_path="rnaseq_data.csv",
        data_type="rnaseq",
        feature_metadata_path="rnaseq_features.csv",
        sample_metadata_path="sample_metadata.csv"
    )
    
    # Load metagenomics dataset
    metagenomics = data_loader.load_dataset(
        data_path="metagenomics_data.csv",
        data_type="metagenomics",
        feature_metadata_path="metagenomics_features.csv",
        sample_metadata_path="sample_metadata.csv"
    )
    
    # Create cross-modality workflow
    workflow = WorkflowFactory.create_cross_modality_workflow([rnaseq, metagenomics])
    
    # Execute workflow
    integrated_data, info = workflow.execute()
    
    # Visualize results
    visualizer = ResultVisualizer()
    
    # Get sample labels for visualization
    labels = rnaseq.sample_metadata['condition'].values
    
    # Plot PCA
    pca_fig = visualizer.plot_pca(integrated_data, labels, "PCA of Integrated RNA-seq and Metagenomics")
    pca_fig.savefig("integrated_pca.png")
    
    # Plot UMAP
    umap_fig = visualizer.plot_umap(integrated_data, labels, "UMAP of Integrated RNA-seq and Metagenomics")
    umap_fig.savefig("integrated_umap.png")
    
    print("Integration complete. Results saved to integrated_pca.png and integrated_umap.png")


# Alternative workflow using config
def config_based_example():
    """Example usage with configuration"""
    # Create configuration
    config = {
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
            "hidden_dims": [128, 64],
            "epochs": 200
        }
    }
    
    # Save configuration
    import json
    with open("integration_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Load datasets
    data_loader = DataLoader()
    
    # Load RNA-seq dataset
    rnaseq = data_loader.load_dataset(
        data_path="rnaseq_data.csv",
        data_type="rnaseq",
        sample_metadata_path="sample_metadata.csv"
    )
    
    # Load epigenomics dataset
    epigenomics = data_loader.load_dataset(
        data_path="epigenomics_data.csv",
        data_type="epigenomics",
        sample_metadata_path="sample_metadata.csv"
    )
    
    # Create workflow from config
    workflow = WorkflowFactory.create_workflow_from_config("integration_config.json")
    
    # Add datasets
    workflow.add_dataset(rnaseq)
    workflow.add_dataset(epigenomics)
    
    # Execute workflow
    integrated_data, info = workflow.execute()
    
    # Visualize results
    visualizer = ResultVisualizer()
    
    # Get sample labels for visualization
    labels = rnaseq.sample_metadata['condition'].values
    
    # Plot PCA
    pca_fig = visualizer.plot_pca(integrated_data, labels, "PCA of Integrated RNA-seq and Epigenomics")
    pca_fig.savefig("integrated_pca.png")
    
    print("Integration complete. Results saved to integrated_pca.png")


if __name__ == "__main__":
    example_usage()
    # Alternatively: config_based_example()

