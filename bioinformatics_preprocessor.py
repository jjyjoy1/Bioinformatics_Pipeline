import numpy as np
import pandas as pd
from scipy import stats
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from abc import ABC, abstractmethod

class BioinformaticsPreprocessor:
    """
    Main orchestrator class for preprocessing various types of bioinformatics data.
    
    This class manages the workflow for different data types and delegates
    specific processing tasks to specialized processor classes.
    """
    
    def __init__(self, data_type=None, logger=None):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_type : str
            Type of bioinformatics data ('rna_seq', 'chip_seq', 'metagenomics', 
            'gwas_eqtl', 'clinical')
        logger : logging.Logger
            Logger for tracking the preprocessing steps
        """
        self.data_type = data_type
        if logger is None:
            self.logger = self._setup_logger()
        else:
            self.logger = logger
            
        # Create appropriate processor based on data type
        if data_type == 'rna_seq':
            self.processor = RNASeqProcessor(logger=self.logger)
        elif data_type == 'chip_seq':
            self.processor = ChIPSeqProcessor(logger=self.logger)
        elif data_type == 'metagenomics':
            self.processor = MetagenomicsProcessor(logger=self.logger)
        elif data_type == 'gwas_eqtl':
            self.processor = GWASProcessor(logger=self.logger)
        elif data_type == 'methyl_seq':  # Add this new condition
            self.processor = MethylSeqProcessor(logger=self.logger)
        elif data_type == 'clinical':
            self.processor = ClinicalProcessor(logger=self.logger)
        else:
            raise ValueError(f"Unsupported data type: {data_type}. Please specify a valid data type.")
            
    def _setup_logger(self):
        """Setup a basic logger if none is provided."""
        logger = logging.getLogger("BioinformaticsPreprocessor")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter features based on specified criteria.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to be filtered
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered data
        """
        self.logger.info(f"Filtering features using method: {method}")
        return self.processor.filter_features(data, method, **kwargs)
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to be normalized
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized data
        """
        self.logger.info(f"Normalizing data using method: {method}")
        return self.processor.normalize(data, method, **kwargs)
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters specific to the imputation method
            
        Returns:
        --------
        pandas.DataFrame
            Data with imputed values
        """
        self.logger.info(f"Imputing missing values using method: {method}")
        return self.processor.impute(data, method, **kwargs)
    
    def process_pipeline(self, data, steps=None, **kwargs):
        """
        Run a complete processing pipeline with multiple steps.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to be processed
        steps : list
            List of processing steps to apply ('filter', 'normalize', 'impute')
        **kwargs : 
            Additional parameters for each step:
            - filter_kwargs: dict, parameters for filtering
            - normalize_kwargs: dict, parameters for normalization
            - impute_kwargs: dict, parameters for imputation
            
        Returns:
        --------
        pandas.DataFrame
            Processed data
        """
        if steps is None:
            steps = ['filter', 'normalize', 'impute']  # Default workflow
            
        result = data.copy()
        
        self.logger.info(f"Starting processing pipeline with steps: {steps}")
        
        for step in steps:
            if step == 'filter':
                filter_kwargs = kwargs.get('filter_kwargs', {})
                result = self.filter_features(result, **filter_kwargs)
            elif step == 'normalize':
                normalize_kwargs = kwargs.get('normalize_kwargs', {})
                result = self.normalize(result, **normalize_kwargs)
            elif step == 'impute':
                impute_kwargs = kwargs.get('impute_kwargs', {})
                result = self.impute(result, **impute_kwargs)
            else:
                self.logger.warning(f"Unknown processing step: {step}")
                
        self.logger.info("Processing pipeline completed successfully")
        return result
    
    # Aliases for backward compatibility
    def preprocess(self, data, method=None, **kwargs):
        """Alias for normalize() to maintain backward compatibility."""
        return self.normalize(data, method, **kwargs)
        
    def fit_transform(self, data, method=None, **kwargs):
        """Alias for normalize() to maintain API compatibility with scikit-learn."""
        return self.normalize(data, method, **kwargs)
        
    def transform(self, data, method=None, **kwargs):
        """Alias for normalize() to maintain API compatibility with scikit-learn."""
        return self.normalize(data, method, **kwargs)


class DataProcessor(ABC):
    """
    Abstract base class for data type-specific processors.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the processor.
        
        Parameters:
        -----------
        logger : logging.Logger
            Logger for tracking the processing steps
        """
        self.logger = logger
    
    @abstractmethod
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter features based on specified criteria.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to be filtered
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered data
        """
        pass
    
    @abstractmethod
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to be normalized
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized data
        """
        pass
    
    @abstractmethod
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in the data.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters specific to the imputation method
            
        Returns:
        --------
        pandas.DataFrame
            Data with imputed values
        """
        pass


class RNASeqProcessor(DataProcessor):
    """
    Processor for RNA-seq data.
    """
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter RNA-seq features.
        
        Methods:
        - 'low_count': Filter genes with counts below a threshold
        - 'low_variance': Filter genes with variance below a threshold
        - 'cpm': Filter genes with CPM below a threshold
        - 'cv': Filter genes based on coefficient of variation
        - 'combined': Apply multiple filtering criteria
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq count data (genes as rows, samples as columns)
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters:
            - min_count: int, minimum count threshold (for 'low_count')
            - min_samples: int/float, minimum number/fraction of samples (for 'low_count', 'cpm')
            - min_variance: float, minimum variance threshold (for 'low_variance')
            - min_cpm: float, minimum CPM threshold (for 'cpm')
            - cv_threshold: float, CV threshold (for 'cv')
            - cv_percentile: float, CV percentile to use as threshold (for 'cv')
            
        Returns:
        --------
        pandas.DataFrame
            Filtered RNA-seq data
        """
        if method is None:
            method = 'low_count'  # Default method
            
        if method == 'low_count':
            return self._filter_low_count(data, **kwargs)
        elif method == 'low_variance':
            return self._filter_low_variance(data, **kwargs)
        elif method == 'cpm':
            return self._filter_cpm(data, **kwargs)
        elif method == 'cv':
            return self._filter_cv(data, **kwargs)
        elif method == 'combined':
            # Apply multiple filtering methods sequentially
            filtered_data = data.copy()
            
            for filter_method, filter_kwargs in kwargs.items():
                if filter_method in ['low_count', 'low_variance', 'cpm', 'cv']:
                    filtered_data = self.filter_features(filtered_data, 
                                                        method=filter_method, 
                                                        **filter_kwargs)
                    
            return filtered_data
        else:
            raise ValueError(f"Unsupported RNA-seq filtering method: {method}")
    
    def _filter_low_count(self, data, min_count=10, min_samples=0.1, **kwargs):
        """
        Filter genes with counts below a threshold.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq count data
        min_count : int
            Minimum count threshold
        min_samples : int or float
            If int: minimum number of samples that must exceed the threshold
            If float (0-1): minimum fraction of samples that must exceed the threshold
        """
        # Convert min_samples from fraction to count if necessary
        if 0 < min_samples < 1:
            min_samples = int(np.ceil(min_samples * data.shape[1]))
            
        # Count samples exceeding the threshold for each gene
        samples_passing = (data >= min_count).sum(axis=1)
        
        # Keep genes with sufficient samples passing the threshold
        keep_genes = samples_passing >= min_samples
        
        filtered_data = data.loc[keep_genes]
        
        self.logger.info(f"Low count filtering: removed {sum(~keep_genes)} out of {len(keep_genes)} genes")
        
        return filtered_data
    
    def _filter_low_variance(self, data, min_variance=0.1, **kwargs):
        """
        Filter genes with variance below a threshold.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq count data
        min_variance : float
            Minimum variance threshold
        """
        # Calculate variance for each gene
        variances = data.var(axis=1)
        
        # Keep genes with variance above the threshold
        keep_genes = variances >= min_variance
        
        filtered_data = data.loc[keep_genes]
        
        self.logger.info(f"Low variance filtering: removed {sum(~keep_genes)} out of {len(keep_genes)} genes")
        
        return filtered_data
    
    def _filter_cpm(self, data, min_cpm=1.0, min_samples=0.1, **kwargs):
        """
        Filter genes with CPM (Counts Per Million) below a threshold.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq count data
        min_cpm : float
            Minimum CPM threshold
        min_samples : int or float
            If int: minimum number of samples that must exceed the threshold
            If float (0-1): minimum fraction of samples that must exceed the threshold
        """
        # Convert min_samples from fraction to count if necessary
        if 0 < min_samples < 1:
            min_samples = int(np.ceil(min_samples * data.shape[1]))
            
        # Calculate CPM for each gene
        library_sizes = data.sum(axis=0)
        cpm = data.div(library_sizes, axis=1) * 1e6
        
        # Count samples exceeding the threshold for each gene
        samples_passing = (cpm >= min_cpm).sum(axis=1)
        
        # Keep genes with sufficient samples passing the threshold
        keep_genes = samples_passing >= min_samples
        
        filtered_data = data.loc[keep_genes]
        
        self.logger.info(f"CPM filtering: removed {sum(~keep_genes)} out of {len(keep_genes)} genes")
        
        return filtered_data
    
    def _filter_cv(self, data, cv_threshold=None, cv_percentile=0.1, **kwargs):
        """
        Filter genes based on coefficient of variation (CV).
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq count data
        cv_threshold : float
            CV threshold, genes with CV below this will be removed
        cv_percentile : float
            If cv_threshold is None, use this percentile to determine the threshold
            (e.g., 0.1 means remove the bottom 10% of genes by CV)
        """
        # Calculate coefficient of variation for each gene
        # CV = standard deviation / mean
        means = data.mean(axis=1)
        stds = data.std(axis=1)
        cv = stds / means
        
        # Handle genes with mean of 0 (CV would be NaN or inf)
        cv = cv.replace([np.inf, -np.inf], np.nan)
        cv = cv.fillna(0)  # Treat genes with 0 mean as having 0 CV
        
        # Determine the threshold if not provided
        if cv_threshold is None:
            cv_threshold = np.nanpercentile(cv, cv_percentile * 100)
            
        # Keep genes with CV above the threshold
        keep_genes = cv >= cv_threshold
        
        filtered_data = data.loc[keep_genes]
        
        self.logger.info(f"CV filtering: removed {sum(~keep_genes)} out of {len(keep_genes)} genes")
        
        return filtered_data
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize RNA-seq data.
        
        Methods:
        - 'log': Simple log2 transformation with pseudocount
        - 'vst': Variance stabilizing transformation (approximation)
        - 'tpm': Transcripts Per Million normalization
        - 'deseq': DESeq2-like size factor normalization (simplified)
        - 'tmm': Trimmed Mean of M-values normalization (simplified)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq count data (genes as rows, samples as columns)
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized RNA-seq data
        """
        if method is None:
            method = 'log'  # Default method
            
        if method == 'log':
            # Log2 transformation with pseudocount
            pseudocount = kwargs.get('pseudocount', 1)
            return pd.DataFrame(np.log2(data + pseudocount), index=data.index, columns=data.columns)
            
        elif method == 'vst':
            # Simplified variance stabilizing transformation
            return pd.DataFrame(np.sqrt(data), index=data.index, columns=data.columns)
            
        elif method == 'tpm':
            # Simple TPM normalization
            gene_lengths = kwargs.get('gene_lengths')
            if gene_lengths is None:
                raise ValueError("Gene lengths must be provided for TPM normalization")
                
            # TPM calculation
            rpk = data.div(gene_lengths, axis=0) * 1000
            scaling_factors = rpk.sum(axis=0) / 1e6
            tpm = rpk.div(scaling_factors, axis=1)
            return tpm
            
        elif method == 'deseq':
            # Simplified DESeq2 normalization
            # Calculate size factors using median ratio method
            geometric_means = stats.gmean(data.values, axis=1)
            size_factors = data.div(geometric_means, axis=0).median(axis=0)
            normalized = data.div(size_factors, axis=1)
            return normalized
            
        elif method == 'tmm':
            # Simplified TMM normalization
            lib_sizes = data.sum(axis=0)
            normalized = data.div(lib_sizes, axis=1) * 1e6  # Simplification to CPM
            return normalized
            
        else:
            raise ValueError(f"Unsupported RNA-seq normalization method: {method}")
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in RNA-seq data.
        
        Methods:
        - 'knn': K-nearest neighbors imputation
        - 'zero': Zero imputation
        - 'mean': Mean imputation
        - 'median': Median imputation
        - 'sciimpute_wrapper': Wrapper for scImpute (requires scImpute package)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            RNA-seq data with missing values (genes as rows, samples as columns)
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters:
            - n_neighbors: int, number of neighbors for KNN imputation
            - weights: str, weight function for KNN ('uniform', 'distance')
            
        Returns:
        --------
        pandas.DataFrame
            RNA-seq data with imputed values
        """
        if method is None:
            method = 'knn'  # Default method
            
        # Check if there are missing values
        if not data.isna().any().any():
            self.logger.info("No missing values found in the data")
            return data
            
        if method == 'knn':
            # KNN imputation
            n_neighbors = kwargs.get('n_neighbors', 5)
            weights = kwargs.get('weights', 'uniform')
            
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputed_values = imputer.fit_transform(data.T).T  # Transpose for sample-based imputation
            
            return pd.DataFrame(imputed_values, index=data.index, columns=data.columns)
            
        elif method == 'zero':
            # Zero imputation
            return data.fillna(0)
            
        elif method == 'mean':
            # Mean imputation (by gene)
            return data.fillna(data.mean(axis=1))
            
        elif method == 'median':
            # Median imputation (by gene)
            return data.fillna(data.median(axis=1))
            
        elif method == 'sciimpute_wrapper':
            # Check if scImpute is installed
            try:
                import rpy2.robjects as robjects
                from rpy2.robjects.packages import importr
                
                # Try to import scImpute
                scimpute = importr('scImpute')
            except (ImportError, ModuleNotFoundError):
                self.logger.error("scImpute imputation requires rpy2 package and R with scImpute installed")
                raise ImportError("scImpute imputation requires rpy2 package and R with scImpute installed")
                
            # Implementation details would depend on scImpute API
            # This is just a placeholder
            self.logger.warning("scImpute wrapper is a placeholder and requires external R package")
            return data
            
        else:
            raise ValueError(f"Unsupported RNA-seq imputation method: {method}")


class ChIPSeqProcessor(DataProcessor):
    """
    Processor for ChIP-seq data.
    """
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter ChIP-seq features.
        
        Methods:
        - 'coverage': Filter peaks based on coverage
        - 'intensity': Filter peaks based on signal intensity
        - 'consistency': Filter peaks based on replicate consistency
        
        Parameters:
        -----------
        data : pandas.DataFrame
            ChIP-seq data (peaks as rows, samples as columns)
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered ChIP-seq data
        """
        if method is None:
            method = 'intensity'  # Default method
            
        # Placeholder for actual implementation
        self.logger.warning(f"ChIP-seq filtering method '{method}' is a placeholder")
        return data
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize ChIP-seq data.
        
        Methods:
        - 'rpm': Reads Per Million normalization
        - 'snr': Signal-to-noise ratio normalization
        
        Parameters:
        -----------
        data : pandas.DataFrame
            ChIP-seq data (peaks as rows, samples as columns)
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized ChIP-seq data
        """
        if method is None:
            method = 'rpm'  # Default method
            
        if method == 'rpm':
            # Reads per million normalization
            lib_sizes = data.sum(axis=0)
            rpm = data.div(lib_sizes, axis=1) * 1e6
            return rpm
            
        elif method == 'snr':
            # Signal-to-noise ratio normalization
            control_samples = kwargs.get('control_samples')
            if control_samples is None:
                raise ValueError("Control samples must be provided for SNR normalization")
                
            # Extract control data
            control_data = data[control_samples]
            treatment_data = data.drop(columns=control_samples)
            
            # Calculate mean of control for each feature
            control_means = control_data.mean(axis=1)
            
            # Calculate signal-to-noise ratio
            snr = treatment_data.div(control_means, axis=0)
            return snr
            
        else:
            raise ValueError(f"Unsupported ChIP-seq normalization method: {method}")
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in ChIP-seq data.
        
        Methods:
        - 'knn': K-nearest neighbors imputation
        - 'zero': Zero imputation
        - 'background': Background level imputation
        
        Parameters:
        -----------
        data : pandas.DataFrame
            ChIP-seq data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters specific to the imputation method
            
        Returns:
        --------
        pandas.DataFrame
            ChIP-seq data with imputed values
        """
        if method is None:
            method = 'knn'  # Default method
            
        # Placeholder for actual implementation
        self.logger.warning(f"ChIP-seq imputation method '{method}' is a placeholder")
        return data


class MetagenomicsProcessor(DataProcessor):
    """
    Processor for metagenomic data.
    """
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter metagenomic features.
        
        Methods:
        - 'prevalence': Filter taxa based on prevalence
        - 'abundance': Filter taxa based on abundance
        - 'diversity': Filter taxa based on contribution to diversity
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Metagenomic data (taxa as rows, samples as columns)
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered metagenomic data
        """
        if method is None:
            method = 'prevalence'  # Default method
            
        # Placeholder for actual implementation
        self.logger.warning(f"Metagenomic filtering method '{method}' is a placeholder")
        return data
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize metagenomic data.
        
        Methods:
        - 'relative': Relative abundance normalization
        - 'clr': Centered log-ratio normalization
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Metagenomic data (taxa as rows, samples as columns)
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized metagenomic data
        """
        if method is None:
            method = 'relative'  # Default method
            
        if method == 'relative':
            # Relative abundance normalization
            sample_totals = data.sum(axis=0)
            relative_abundance = data.div(sample_totals, axis=1)
            return relative_abundance
            
        elif method == 'clr':
            # Centered log-ratio normalization
            pseudocount = kwargs.get('pseudocount', 1e-6)
            
            # Add pseudocount and calculate geometric mean for each sample
            data_with_pseudocount = data + pseudocount
            
            # Calculate CLR
            clr = data_with_pseudocount.apply(lambda x: np.log(x / stats.gmean(x)), axis=0)
            return clr
            
        else:
            raise ValueError(f"Unsupported metagenomic normalization method: {method}")
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in metagenomic data.
        
        Methods:
        - 'knn': K-nearest neighbors imputation
        - 'zero': Zero imputation
        - 'pseudocount': Pseudocount addition
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Metagenomic data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters specific to the imputation method
            
        Returns:
        --------
        pandas.DataFrame
            Metagenomic data with imputed values
        """
        if method is None:
            method = 'zero'  # Default method
            
        # Placeholder for actual implementation
        self.logger.warning(f"Metagenomic imputation method '{method}' is a placeholder")
        return data


class GWASProcessor(DataProcessor):
    """
    Processor for GWAS/eQTL data.
    """
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter GWAS/eQTL features.
        
        Methods:
        - 'maf': Filter variants based on minor allele frequency
        - 'hwe': Filter variants based on Hardy-Weinberg equilibrium
        - 'missing': Filter variants based on missing data rate
        - 'ld': Filter variants based on linkage disequilibrium
        
        Parameters:
        -----------
        data : pandas.DataFrame
            GWAS/eQTL data (variants as rows, samples as columns)
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered GWAS/eQTL data
        """
        if method is None:
            method = 'maf'  # Default method
            
        # Placeholder for actual implementation
        self.logger.warning(f"GWAS/eQTL filtering method '{method}' is a placeholder")
        return data
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize GWAS/eQTL data.
        
        Methods:
        - 'standardize': Z-score standardization
        - 'rank': Rank-based normalization
        
        Parameters:
        -----------
        data : pandas.DataFrame
            GWAS/eQTL data (variants as rows, samples as columns)
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized GWAS/eQTL data
        """
        if method is None:
            method = 'standardize'  # Default method
            
        if method == 'standardize':
            # Z-score standardization
            centered = data.sub(data.mean(axis=0), axis=1)
            standardized = centered.div(data.std(axis=0), axis=1)
            return standardized
            
        elif method == 'rank':
            # Rank-based normalization
            ranked = data.rank(axis=0)
            total = data.shape[0]
            
            # Scale to [0,1]
            normalized = (ranked - 0.5) / total
            
            # Optional inverse normal transformation
            if kwargs.get('inverse_normal', False):
                from scipy.stats import norm
                normalized = normalized.apply(lambda x: norm.ppf(x))
                
            return normalized
            
        else:
            raise ValueError(f"Unsupported GWAS/eQTL normalization method: {method}")
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in GWAS/eQTL data.
        
        Methods:
        - 'genotype': Genotype imputation based on linkage disequilibrium
        - 'mean': Mean imputation
        - 'mode': Mode imputation for categorical variants
        
        Parameters:
        -----------
        data : pandas.DataFrame
            GWAS/eQTL data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters specific to the imputation method
            
        Returns:
        --------
        pandas.DataFrame
            GWAS/eQTL data with imputed values
        """
        if method is None:
            method = 'mean'  # Default method
            
        # Placeholder for actual implementation
        self.logger.warning(f"GWAS/eQTL imputation method '{method}' is a placeholder")
        return data


class ClinicalProcessor(DataProcessor):
    """
    Processor for clinical metadata.
    """
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter clinical features.
        
        Methods:
        - 'variance': Filter numerical features based on variance
        - 'frequency': Filter categorical features based on frequency
        - 'correlation': Filter features based on correlation
        - 'missing': Filter features with high percentage of missing values
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Clinical data (samples as rows, features as columns)
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered clinical data
        """
        if method is None:
            method = 'variance'  # Default method
            
        if method == 'variance':
            # Filter numerical features based on variance
            threshold = kwargs.get('threshold', 0.01)
            
            # Identify numerical columns
            num_cols = data.select_dtypes(include=np.number).columns
            
            if len(num_cols) == 0:
                self.logger.warning("No numerical features found for variance filtering")
                return data
                
            # Calculate variances
            variances = data[num_cols].var()
            
            # Filter low variance features
            keep_features = variances[variances >= threshold].index
            drop_features = [col for col in num_cols if col not in keep_features]
            
            if drop_features:
                self.logger.info(f"Variance filtering: removed {len(drop_features)} features")
                return data.drop(columns=drop_features)
            else:
                return data
                
        elif method == 'frequency':
            # Filter categorical features based on frequency
            min_frequency = kwargs.get('min_frequency', 0.05)
            
            # Identify categorical columns
            cat_cols = data.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) == 0:
                self.logger.warning("No categorical features found for frequency filtering")
                return data
                
            drop_features = []
            
            for col in cat_cols:
                # Calculate frequency of each category
                value_counts = data[col].value_counts(normalize=True)
                
                # Check if any category has frequency below threshold
                if (value_counts < min_frequency).any():
                    drop_features.append(col)
                    
            if drop_features:
                self.logger.info(f"Frequency filtering: removed {len(drop_features)} features")
                return data.drop(columns=drop_features)
            else:
                return data
                
        elif method == 'correlation':
            # Filter features based on correlation
            threshold = kwargs.get('threshold', 0.9)
            
            # Identify numerical columns
            num_cols = data.select_dtypes(include=np.number).columns
            
            if len(num_cols) == 0:
                self.logger.warning("No numerical features found for correlation filtering")
                return data
                
            # Calculate correlation matrix
            corr_matrix = data[num_cols].corr().abs()
            
            # Find features to drop
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            drop_features = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            if drop_features:
                self.logger.info(f"Correlation filtering: removed {len(drop_features)} features")
                return data.drop(columns=drop_features)
            else:
                return data
                
        elif method == 'missing':
            # Filter features with high percentage of missing values
            threshold = kwargs.get('threshold', 0.2)
            
            # Calculate percentage of missing values for each feature
            missing_percentage = data.isna().mean()
            
            # Find features to drop
            drop_features = missing_percentage[missing_percentage > threshold].index.tolist()
            
            if drop_features:
                self.logger.info(f"Missing value filtering: removed {len(drop_features)} features")
                return data.drop(columns=drop_features)
            else:
                return data
                
        else:
            raise ValueError(f"Unsupported clinical feature filtering method: {method}")
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize clinical data.
        
        Methods:
        - 'auto': Automatically detect and process different variable types
        - 'numerical': Scale numerical features
        - 'categorical': Encode categorical features
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Clinical data (samples as rows, features as columns)
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters:
            - num_scaling: str, numerical scaling method ('standard', 'minmax', 'robust')
            - cat_encoding: str, categorical encoding method ('onehot', 'dummy', 'label')
            
        Returns:
        --------
        pandas.DataFrame
            Normalized clinical data
        """
        if method is None:
            method = 'auto'  # Default method
            
        # Get parameters from kwargs
        num_scaling = kwargs.get('num_scaling', 'standard')
        cat_encoding = kwargs.get('cat_encoding', 'onehot')
        
        # Make a copy to avoid modifying the original
        processed_data = data.copy()
        
        if method == 'auto':
            # Automatically detect and process variable types
            num_cols = processed_data.select_dtypes(include=np.number).columns
            cat_cols = processed_data.select_dtypes(include=['object', 'category']).columns
            
            # Process numerical features
            if len(num_cols) > 0:
                processed_data[num_cols] = self._normalize_numerical(processed_data[num_cols], method=num_scaling)
                
            # Process categorical features
            if len(cat_cols) > 0:
                encoded_data = self._normalize_categorical(processed_data[cat_cols], method=cat_encoding)
                
                # Drop original categorical columns and add encoded ones
                processed_data = processed_data.drop(columns=cat_cols)
                processed_data = pd.concat([processed_data, encoded_data], axis=1)
                
            return processed_data
            
        elif method == 'numerical':
            # Process only numerical features
            num_cols = processed_data.select_dtypes(include=np.number).columns
            
            if len(num_cols) > 0:
                processed_data[num_cols] = self._normalize_numerical(processed_data[num_cols], method=num_scaling)
                
            return processed_data
            
        elif method == 'categorical':
            # Process only categorical features
            cat_cols = processed_data.select_dtypes(include=['object', 'category']).columns
            
            if len(cat_cols) > 0:
                encoded_data = self._normalize_categorical(processed_data[cat_cols], method=cat_encoding)
                
                # Drop original categorical columns and add encoded ones
                processed_data = processed_data.drop(columns=cat_cols)
                processed_data = pd.concat([processed_data, encoded_data], axis=1)
                
            return processed_data
            
        else:
            raise ValueError(f"Unsupported clinical data normalization method: {method}")
    
    def _normalize_numerical(self, data, method='standard'):
        """
        Scale numerical features.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Numerical data
        method : str
            Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
        --------
        pandas.DataFrame
            Scaled numerical data
        """
        if method == 'standard':
            # Standardization (z-score normalization)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(data)
            return pd.DataFrame(scaled, index=data.index, columns=data.columns)
            
        elif method == 'minmax':
            # Min-max scaling to [0,1]
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(data)
            return pd.DataFrame(scaled, index=data.index, columns=data.columns)
            
        elif method == 'robust':
            # Robust scaling using quantiles
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            scaled = scaler.fit_transform(data)
            return pd.DataFrame(scaled, index=data.index, columns=data.columns)
            
        else:
            raise ValueError(f"Unsupported numerical scaling method: {method}")
    
    def _normalize_categorical(self, data, method='onehot'):
        """
        Encode categorical features.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Categorical data
        method : str
            Encoding method ('onehot', 'dummy', 'label')
            
        Returns:
        --------
        pandas.DataFrame
            Encoded categorical data
        """
        if method == 'onehot':
            # One-hot encoding
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded = encoder.fit_transform(data)
            
            # Create feature names
            feature_names = []
            for i, col in enumerate(data.columns):
                for category in encoder.categories_[i]:
                    feature_names.append(f"{col}_{category}")
                    
            return pd.DataFrame(encoded, index=data.index, columns=feature_names)
            
        elif method == 'dummy':
            # Pandas get_dummies (similar to one-hot but drops first category)
            return pd.get_dummies(data, drop_first=True)
            
        elif method == 'label':
            # Label encoding (convert categories to integers)
            from sklearn.preprocessing import LabelEncoder
            encoded_data = pd.DataFrame(index=data.index)
            
            for col in data.columns:
                encoder = LabelEncoder()
                encoded_data[col] = encoder.fit_transform(data[col])
                
            return encoded_data
            
        else:
            raise ValueError(f"Unsupported categorical encoding method: {method}")
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in clinical data.
        
        Methods:
        - 'auto': Automatically select imputation method based on data type
        - 'mean': Mean imputation for numerical features
        - 'median': Median imputation for numerical features
        - 'mode': Mode imputation for categorical features
        - 'knn': K-nearest neighbors imputation
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Clinical data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters:
            - n_neighbors: int, number of neighbors for KNN imputation
            
        Returns:
        --------
        pandas.DataFrame
            Clinical data with imputed values
        """
        if method is None:
            method = 'auto'  # Default method
            
        # Check if there are missing values
        if not data.isna().any().any():
            self.logger.info("No missing values found in the data")
            return data
            
        # Make a copy to avoid modifying the original
        imputed_data = data.copy()
        
        if method == 'auto':
            # Automatically select imputation method based on data type
            num_cols = imputed_data.select_dtypes(include=np.number).columns
            cat_cols = imputed_data.select_dtypes(include=['object', 'category']).columns
            
            # Impute numerical features with mean
            for col in num_cols:
                if imputed_data[col].isna().any():
                    imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mean())
                    
            # Impute categorical features with mode
            for col in cat_cols:
                if imputed_data[col].isna().any():
                    imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mode()[0])
                    
            return imputed_data
            
        elif method == 'mean':
            # Mean imputation for numerical features
            num_cols = imputed_data.select_dtypes(include=np.number).columns
            
            for col in num_cols:
                if imputed_data[col].isna().any():
                    imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mean())
                    
            return imputed_data
            
        elif method == 'median':
            # Median imputation for numerical features
            num_cols = imputed_data.select_dtypes(include=np.number).columns
            
            for col in num_cols:
                if imputed_data[col].isna().any():
                    imputed_data[col] = imputed_data[col].fillna(imputed_data[col].median())
                    
            return imputed_data
            
        elif method == 'mode':
            # Mode imputation for all features
            for col in imputed_data.columns:
                if imputed_data[col].isna().any():
                    imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mode()[0])
                    
            return imputed_data
            
        elif method == 'knn':
            # KNN imputation
            n_neighbors = kwargs.get('n_neighbors', 5)
            
            # Separate numerical and categorical columns
            num_cols = imputed_data.select_dtypes(include=np.number).columns
            cat_cols = imputed_data.select_dtypes(include=['object', 'category']).columns
            
            # Impute numerical features with KNN
            if len(num_cols) > 0 and imputed_data[num_cols].isna().any().any():
                imputer = KNNImputer(n_neighbors=n_neighbors)
                imputed_num = imputer.fit_transform(imputed_data[num_cols])
                imputed_data[num_cols] = pd.DataFrame(imputed_num, index=imputed_data.index, columns=num_cols)
                
            # Impute categorical features with mode (KNN doesn't work directly with categorical)
            for col in cat_cols:
                if imputed_data[col].isna().any():
                    imputed_data[col] = imputed_data[col].fillna(imputed_data[col].mode()[0])
                    
            return imputed_data
            
        else:
            raise ValueError(f"Unsupported clinical data imputation method: {method}")


class MethylSeqProcessor(DataProcessor):
    """
    Processor for DNA methylation sequencing data.
    
    Handles processing of various methylation data types:
    - Whole Genome Bisulfite Sequencing (WGBS)
    - Reduced Representation Bisulfite Sequencing (RRBS)
    - Methylation arrays (e.g., Illumina 450K, EPIC)
    """
    
    def filter_features(self, data, method=None, **kwargs):
        """
        Filter methylation features.
        
        Methods:
        - 'coverage': Filter CpG sites based on coverage/detection
        - 'variance': Filter CpG sites based on methylation variance
        - 'pca_outliers': Remove outlier CpG sites based on PCA
        - 'nonvariable': Remove non-variable CpG sites
        - 'context': Filter CpG sites based on genomic context (CpG islands, shores, etc.)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Methylation data (CpG sites as rows, samples as columns)
        method : str
            Filtering method to use
        **kwargs : 
            Additional parameters specific to the filtering method
            
        Returns:
        --------
        pandas.DataFrame
            Filtered methylation data
        """
        if method is None:
            method = 'coverage'  # Default method
            
        if method == 'coverage':
            # Filter CpG sites based on coverage
            min_coverage = kwargs.get('min_coverage', 10)
            min_samples = kwargs.get('min_samples', 0.7)  # Proportion of samples
            coverage_data = kwargs.get('coverage_data')
            
            if coverage_data is None:
                self.logger.warning("Coverage data not provided, assuming methylation beta values")
                # For beta values, we can only filter based on NAs
                return self._filter_detection(data, min_samples)
            
            # Convert min_samples from fraction to count if necessary
            if 0 < min_samples < 1:
                min_samples = int(np.ceil(min_samples * data.shape[1]))
                
            # Count samples exceeding the coverage threshold for each CpG site
            samples_passing = (coverage_data >= min_coverage).sum(axis=1)
            
            # Keep CpG sites with sufficient samples passing the threshold
            keep_sites = samples_passing >= min_samples
            
            filtered_data = data.loc[keep_sites]
            
            self.logger.info(f"Coverage filtering: removed {sum(~keep_sites)} out of {len(keep_sites)} CpG sites")
            
            return filtered_data
            
        elif method == 'variance':
            # Filter CpG sites based on methylation variance
            min_variance = kwargs.get('min_variance', 0.001)
            percentile = kwargs.get('percentile')
            
            # Calculate variance for each CpG site
            variances = data.var(axis=1)
            
            # If percentile is specified, use it to determine the threshold
            if percentile is not None:
                min_variance = np.percentile(variances, percentile)
                
            # Keep CpG sites with variance above the threshold
            keep_sites = variances >= min_variance
            
            filtered_data = data.loc[keep_sites]
            
            self.logger.info(f"Variance filtering: removed {sum(~keep_sites)} out of {len(keep_sites)} CpG sites")
            
            return filtered_data
            
        elif method == 'pca_outliers':
            # Remove outlier CpG sites based on PCA
            from sklearn.decomposition import PCA
            n_components = kwargs.get('n_components', 10)
            z_threshold = kwargs.get('z_threshold', 3)
            
            # Transpose for PCA (samples as rows, CpG sites as columns)
            data_t = data.T
            
            # Impute missing values for PCA (simple mean imputation)
            data_t_imputed = data_t.fillna(data_t.mean())
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            pca.fit(data_t_imputed)
            
            # Calculate loadings
            loadings = pca.components_
            
            # Calculate Z-scores for loadings
            z_scores = np.abs(stats.zscore(loadings, axis=1))
            
            # Find outlier CpG sites
            outliers = np.any(z_scores > z_threshold, axis=0)
            
            # Keep non-outlier CpG sites
            keep_sites = ~outliers
            
            filtered_data = data.loc[data.index[keep_sites]]
            
            self.logger.info(f"PCA outlier filtering: removed {sum(outliers)} out of {len(outliers)} CpG sites")
            
            return filtered_data
            
        elif method == 'nonvariable':
            # Remove non-variable CpG sites
            min_range = kwargs.get('min_range', 0.1)
            min_samples = kwargs.get('min_samples', 0.7)  # Proportion of samples
            
            # Convert min_samples from fraction to count if necessary
            if 0 < min_samples < 1:
                min_samples = int(np.ceil(min_samples * data.shape[1]))
                
            # Calculate range for each CpG site
            ranges = data.max(axis=1) - data.min(axis=1)
            
            # Keep CpG sites with range above the threshold
            keep_sites = ranges >= min_range
            
            filtered_data = data.loc[keep_sites]
            
            self.logger.info(f"Non-variable filtering: removed {sum(~keep_sites)} out of {len(keep_sites)} CpG sites")
            
            return filtered_data
            
        elif method == 'context':
            # Filter CpG sites based on genomic context
            context_type = kwargs.get('context_type', 'islands')  # 'islands', 'shores', 'shelves', 'open_sea'
            annotation = kwargs.get('annotation')
            
            if annotation is None:
                raise ValueError("Annotation data must be provided for context filtering")
                
            # Filter based on the specified context
            keep_sites = annotation[context_type]
            
            filtered_data = data.loc[keep_sites]
            
            self.logger.info(f"Context filtering ({context_type}): kept {sum(keep_sites)} out of {len(keep_sites)} CpG sites")
            
            return filtered_data
            
        else:
            raise ValueError(f"Unsupported methylation filtering method: {method}")
    
    def _filter_detection(self, data, min_samples=0.7):
        """
        Filter CpG sites based on detection/missing values.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Methylation data
        min_samples : int or float
            If int: minimum number of samples with non-missing values
            If float (0-1): minimum fraction of samples with non-missing values
        """
        # Convert min_samples from fraction to count if necessary
        if 0 < min_samples < 1:
            min_samples = int(np.ceil(min_samples * data.shape[1]))
            
        # Count non-missing samples for each CpG site
        samples_detected = data.notna().sum(axis=1)
        
        # Keep CpG sites with sufficient detected samples
        keep_sites = samples_detected >= min_samples
        
        filtered_data = data.loc[keep_sites]
        
        self.logger.info(f"Detection filtering: removed {sum(~keep_sites)} out of {len(keep_sites)} CpG sites")
        
        return filtered_data
    
    def normalize(self, data, method=None, **kwargs):
        """
        Normalize methylation data.
        
        Methods:
        - 'beta': Convert to beta values (0-1 scale)
        - 'm_value': Convert to M-values (logit transform)
        - 'quantile': Quantile normalization
        - 'bmiq': Beta Mixture Quantile normalization
        - 'swan': Subset-quantile Within Array Normalization
        - 'funnorm': Functional normalization
        - 'noob': Normal-exponential out-of-band
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Methylation data (CpG sites as rows, samples as columns)
        method : str
            Normalization method to use
        **kwargs : 
            Additional parameters specific to the normalization method
            
        Returns:
        --------
        pandas.DataFrame
            Normalized methylation data
        """
        if method is None:
            method = 'beta'  # Default method
            
        if method == 'beta':
            # Convert to beta values (0-1 scale)
            methylated = kwargs.get('methylated')
            unmethylated = kwargs.get('unmethylated')
            offset = kwargs.get('offset', 100)
            
            if methylated is None or unmethylated is None:
                self.logger.warning("Methylated and unmethylated data not provided, assuming data is already in beta values")
                return data
                
            # Calculate beta values: beta = M / (M + U + offset)
            beta_values = methylated / (methylated + unmethylated + offset)
            
            # Clip values to [0, 1]
            beta_values = np.clip(beta_values, 0, 1)
            
            return pd.DataFrame(beta_values, index=data.index, columns=data.columns)
            
        elif method == 'm_value':
            # Convert to M-values (logit transform)
            beta_values = kwargs.get('beta_values', data)
            epsilon = kwargs.get('epsilon', 1e-5)
            
            # Clip beta values to avoid log(0) or log(1)
            beta_clipped = np.clip(beta_values, epsilon, 1 - epsilon)
            
            # Calculate M-values: M = log2(beta / (1 - beta))
            m_values = np.log2(beta_clipped / (1 - beta_clipped))
            
            return pd.DataFrame(m_values, index=data.index, columns=data.columns)
            
        elif method == 'quantile':
            # Quantile normalization
            from sklearn.preprocessing import QuantileTransformer
            
            # Transpose for normalization (samples as rows, CpG sites as columns)
            data_t = data.T
            
            # Handle missing values (temporary replacement with median)
            has_na = data_t.isna().any().any()
            if has_na:
                medians = data_t.median()
                data_t_filled = data_t.fillna(medians)
            else:
                data_t_filled = data_t
                
            # Apply quantile normalization
            transformer = QuantileTransformer(output_distribution='normal')
            normalized_data_t = transformer.fit_transform(data_t_filled)
            
            # Convert back to original shape
            normalized_data = pd.DataFrame(normalized_data_t.T, index=data.index, columns=data.columns)
            
            # Restore missing values if needed
            if has_na:
                mask = data.isna()
                normalized_data[mask] = np.nan
                
            return normalized_data
            
        elif method in ['bmiq', 'swan', 'funnorm', 'noob']:
            # These methods typically require specialized libraries like 'minfi' in R
            self.logger.warning(f"Normalization method '{method}' requires external R libraries. Consider using rpy2 to interface with R packages.")
            
            # Placeholder for R integration
            try:
                # Check if rpy2 is available
                import rpy2.robjects as robjects
                from rpy2.robjects import pandas2ri
                
                # Example for BMIQ (would need to be expanded)
                if method == 'bmiq':
                    # Code to call R's BMIQ function would go here
                    pass
                    
                self.logger.warning(f"R integration for {method} normalization is not fully implemented")
                return data
                
            except ImportError:
                self.logger.error(f"rpy2 package is required for {method} normalization")
                raise ImportError(f"rpy2 package is required for {method} normalization")
                
        else:
            raise ValueError(f"Unsupported methylation normalization method: {method}")
    
    def impute(self, data, method=None, **kwargs):
        """
        Impute missing values in methylation data.
        
        Methods:
        - 'knn': K-nearest neighbors imputation
        - 'mean': Mean imputation
        - 'median': Median imputation
        - 'similar_cpg': Imputation based on similar CpG sites
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Methylation data with missing values
        method : str
            Imputation method to use
        **kwargs : 
            Additional parameters specific to the imputation method
            
        Returns:
        --------
        pandas.DataFrame
            Methylation data with imputed values
        """
        if method is None:
            method = 'knn'  # Default method
            
        # Check if there are missing values
        if not data.isna().any().any():
            self.logger.info("No missing values found in the data")
            return data
            
        if method == 'knn':
            # KNN imputation
            n_neighbors = kwargs.get('n_neighbors', 5)
            weights = kwargs.get('weights', 'uniform')
            
            imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            imputed_values = imputer.fit_transform(data.T).T  # Transpose for sample-based imputation
            
            return pd.DataFrame(imputed_values, index=data.index, columns=data.columns)
            
        elif method == 'mean':
            # Mean imputation (by CpG site)
            return data.fillna(data.mean(axis=1))
            
        elif method == 'median':
            # Median imputation (by CpG site)
            return data.fillna(data.median(axis=1))
            
        elif method == 'similar_cpg':
            # Imputation based on similar CpG sites
            n_neighbors = kwargs.get('n_neighbors', 10)
            
            # Create a copy to avoid modifying the original
            imputed_data = data.copy()
            
            # Find CpG sites with missing values
            sites_with_na = data.isna().any(axis=1)
            sites_to_impute = data.index[sites_with_na]
            
            # For each CpG site with missing values
            for site in sites_to_impute:
                # Samples with missing values for this site
                samples_with_na = data.loc[site].isna()
                
                if samples_with_na.all():
                    # If all samples are missing, use median of all sites
                    imputed_data.loc[site, samples_with_na] = data.median(axis=0)[samples_with_na]
                    continue
                    
                # Compute correlation with other CpG sites
                # (using only samples where the target site has values)
                site_data = data.loc[site, ~samples_with_na]
                other_sites = data.index[~sites_with_na]  # Use only complete CpG sites
                
                if len(other_sites) == 0:
                    # If no complete sites, use the mean/median
                    imputed_data.loc[site, samples_with_na] = data.loc[site, ~samples_with_na].mean()
                    continue
                    
                # Compute correlations
                correlations = {}
                for other_site in other_sites:
                    other_data = data.loc[other_site, ~samples_with_na]
                    correlations[other_site] = np.corrcoef(site_data, other_data)[0, 1]
                    
                # Sort by absolute correlation
                sorted_sites = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Select top n most correlated CpG sites
                top_sites = [s[0] for s in sorted_sites[:n_neighbors]]
                
                # Impute missing values based on similar CpG sites
                for sample in data.columns[samples_with_na]:
                    # Get values of similar CpG sites for this sample
                    similar_values = data.loc[top_sites, sample].dropna()
                    
                    if len(similar_values) > 0:
                        # Use weighted average based on correlation
                        weights = np.array([abs(correlations[s]) for s in similar_values.index])
                        weights = weights / weights.sum()  # Normalize weights
                        imputed_data.loc[site, sample] = np.sum(similar_values * weights)
                    else:
                        # If no similar values available, use mean of this CpG site
                        imputed_data.loc[site, sample] = data.loc[site, ~samples_with_na].mean()
                        
            return imputed_data
            
        else:
            raise ValueError(f"Unsupported methylation imputation method: {method}")



# Example usage
if __name__ == "__main__":
    # Set up a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Example RNA-seq data
    np.random.seed(42)
    n_genes, n_samples = 100, 20
    rna_data = pd.DataFrame(np.random.negative_binomial(10, 0.5, size=(n_genes, n_samples)))
    rna_data.index = [f"gene_{i}" for i in range(n_genes)]
    rna_data.columns = [f"sample_{i}" for i in range(n_samples)]
    
    # Create the preprocessor
    preprocessor = BioinformaticsPreprocessor(data_type='rna_seq', logger=logger)
    
    # Example 1: Filter low-expressed genes
    filtered_data = preprocessor.filter_features(rna_data, method='low_count', min_count=5, min_samples=0.3)
    
    # Example 2: Normalize using log transformation
    normalized_data = preprocessor.normalize(filtered_data, method='log', pseudocount=1)
    
    # Example 3: Full pipeline
    pipeline_result = preprocessor.process_pipeline(
        rna_data,
        steps=['filter', 'normalize'],
        filter_kwargs={'method': 'low_count', 'min_count': 5, 'min_samples': 0.3},
        normalize_kwargs={'method': 'log', 'pseudocount': 1}
    )
    
    print(f"Original data shape: {rna_data.shape}")
    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Normalized data shape: {normalized_data.shape}")
    print(f"Pipeline result shape: {pipeline_result.shape}")

    #Example of Methlation data preprocessing
    # Create synthetic beta values (0-1 range)
    beta_values = np.random.beta(2, 5, size=(n_cpgs, n_samples))
    methyl_data = pd.DataFrame(beta_values)
    methyl_data.index = [f"cg{i:08d}" for i in range(n_cpgs)]
    methyl_data.columns = [f"sample_{i}" for i in range(n_samples)]
    
    # Add some missing values
    missing_mask = np.random.random(size=(n_cpgs, n_samples)) < 0.05
    methyl_data[missing_mask] = np.nan
    
    # Create coverage data (for filtering)
    coverage_data = np.random.negative_binomial(10, 0.5, size=(n_cpgs, n_samples))
    coverage_data = pd.DataFrame(coverage_data, index=methyl_data.index, columns=methyl_data.columns)
    
    # Create the preprocessor
    preprocessor = BioinformaticsPreprocessor(data_type='methyl_seq')
    
    # Example 1: Filter based on coverage
    filtered_data = preprocessor.filter_features(
        methyl_data,
        method='coverage',
        min_coverage=10,
        min_samples=0.8,
        coverage_data=coverage_data
    )
    
    # Example 2: Convert to M-values
    m_values = preprocessor.normalize(
        filtered_data,
        method='m_value',
        epsilon=1e-5
    )
    
    # Example 3: Impute missing values
    imputed_data = preprocessor.impute(
        m_values,
        method='knn',
        n_neighbors=5
    )
    
    # Example 4: Full processing pipeline
    pipeline_result = preprocessor.process_pipeline(
        methyl_data,
        steps=['filter', 'normalize', 'impute'],
        filter_kwargs={
            'method': 'variance',
            'min_variance': 0.001
        },
        normalize_kwargs={
            'method': 'm_value'
        },
        impute_kwargs={
            'method': 'similar_cpg',
            'n_neighbors': 10
        }
    )
    
    print(f"Original data shape: {methyl_data.shape}")
    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Imputed data shape: {imputed_data.shape}")
    print(f"Pipeline result shape: {pipeline_result.shape}")
