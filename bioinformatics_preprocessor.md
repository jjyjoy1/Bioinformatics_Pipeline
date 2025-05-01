# Bioinformatics Preprocessor

A comprehensive Python framework for preprocessing and normalizing various types of bioinformatics data matrices.

## Overview

Bioinformatics Preprocessor is a flexible, modular framework designed to handle the preprocessing, normalization, and imputation of different types of biological data. This tool converts raw values from various experimental platforms into normalized values suitable for machine learning and deep learning models.

## Features

- **Modular Design**: Separate processors for different data types
- **Comprehensive Functionality**: Filtering, normalization, and imputation for each data type
- **Scikit-learn Compatible**: Follow fit/transform API patterns
- **Extensible Architecture**: Easy to add new methods or data types
- **Efficient Processing**: Optimized for large datasets

## Supported Data Types

- **RNA-seq**: Gene expression data from RNA sequencing
- **ChIP-seq**: Chromatin immunoprecipitation sequencing data
- **Metagenomics**: Microbial community profiling data
- **GWAS/eQTL**: Genome-wide association and expression quantitative trait loci data
- **Clinical**: Patient metadata including categorical and numerical variables
- **MethylSeq**: DNA methylation data from bisulfite sequencing or arrays


## Quick Start

```python
import pandas as pd
from bioinformatics_preprocessor import BioinformaticsPreprocessor

# Load your data
data = pd.read_csv("rna_seq_counts.csv", index_col=0)

# Initialize the preprocessor for RNA-seq data
preprocessor = BioinformaticsPreprocessor(data_type='rna_seq')

# Process the data with a full pipeline
processed_data = preprocessor.process_pipeline(
    data,
    steps=['filter', 'normalize', 'impute'],
    filter_kwargs={'method': 'low_count', 'min_count': 10, 'min_samples': 0.2},
    normalize_kwargs={'method': 'log', 'pseudocount': 1},
    impute_kwargs={'method': 'knn', 'n_neighbors': 5}
)
```

## Detailed Documentation

### RNA-seq Data Processing

#### Filtering Methods

- **low_count**: Remove genes with low expression
- **low_variance**: Remove genes with low variance
- **cpm**: Filter based on counts per million
- **cv**: Filter based on coefficient of variation

#### Normalization Methods

- **log**: Log2 transformation with pseudocount
- **vst**: Variance stabilizing transformation
- **tpm**: Transcripts Per Million normalization
- **deseq**: DESeq2-like size factor normalization
- **tmm**: Trimmed Mean of M-values normalization

#### Imputation Methods

- **knn**: K-nearest neighbors imputation
- **mean**: Mean imputation by gene
- **median**: Median imputation by gene
- **zero**: Zero imputation
- **sciimpute_wrapper**: Wrapper for scImpute (requires R package)

### ChIP-seq Data Processing

#### Filtering Methods

- **coverage**: Filter peaks based on coverage
- **intensity**: Filter peaks based on signal intensity
- **consistency**: Filter peaks based on replicate consistency

#### Normalization Methods

- **rpm**: Reads Per Million normalization
- **snr**: Signal-to-noise ratio normalization

#### Imputation Methods

- **knn**: K-nearest neighbors imputation
- **zero**: Zero imputation
- **background**: Background level imputation

### Metagenomics Data Processing

#### Filtering Methods

- **prevalence**: Filter taxa based on prevalence
- **abundance**: Filter taxa based on abundance
- **diversity**: Filter taxa based on contribution to diversity

#### Normalization Methods

- **relative**: Relative abundance normalization
- **clr**: Centered log-ratio normalization

#### Imputation Methods

- **knn**: K-nearest neighbors imputation
- **zero**: Zero imputation
- **pseudocount**: Pseudocount addition

### GWAS/eQTL Data Processing

#### Filtering Methods

- **maf**: Filter variants based on minor allele frequency
- **hwe**: Filter variants based on Hardy-Weinberg equilibrium
- **missing**: Filter variants based on missing data rate
- **ld**: Filter variants based on linkage disequilibrium

#### Normalization Methods

- **standardize**: Z-score standardization
- **rank**: Rank-based normalization

#### Imputation Methods

- **genotype**: Genotype imputation based on linkage disequilibrium
- **mean**: Mean imputation
- **mode**: Mode imputation for categorical variants

### Clinical Data Processing

#### Filtering Methods

- **variance**: Filter numerical features based on variance
- **frequency**: Filter categorical features based on frequency
- **correlation**: Filter features based on correlation
- **missing**: Filter features with high percentage of missing values

#### Normalization Methods

- **auto**: Automatically detect and process different variable types
- **numerical**: Scale numerical features
- **categorical**: Encode categorical features

#### Imputation Methods

- **auto**: Automatically select imputation method based on data type
- **mean**: Mean imputation for numerical features
- **median**: Median imputation for numerical features
- **mode**: Mode imputation for categorical features
- **knn**: K-nearest neighbors imputation

### MethylSeq Data Processing

#### Filtering Methods

- **coverage**: Filter CpG sites based on coverage/detection
- **variance**: Filter CpG sites based on methylation variance
- **pca_outliers**: Remove outlier CpG sites based on PCA
- **nonvariable**: Remove non-variable CpG sites
- **context**: Filter CpG sites based on genomic context

#### Normalization Methods

- **beta**: Convert to beta values (0-1 scale)
- **m_value**: Convert to M-values (logit transform)
- **quantile**: Quantile normalization
- **bmiq**: Beta Mixture Quantile normalization
- **swan**: Subset-quantile Within Array Normalization
- **funnorm**: Functional normalization
- **noob**: Normal-exponential out-of-band

#### Imputation Methods

- **knn**: K-nearest neighbors imputation
- **mean**: Mean imputation
- **median**: Median imputation
- **similar_cpg**: Imputation based on similar CpG sites

## Advanced Usage

### Custom Processing Pipeline

```python
import pandas as pd
import numpy as np
from bioinformatics_preprocessor import BioinformaticsPreprocessor

# Load RNA-seq count data
counts = pd.read_csv("counts.csv", index_col=0)

# Initialize preprocessor
preprocessor = BioinformaticsPreprocessor(data_type='rna_seq')

# Step 1: Filter low-count genes
filtered_data = preprocessor.filter_features(
    counts, 
    method='low_count',
    min_count=10,
    min_samples=0.3
)

# Step 2: Calculate TPM values
gene_lengths = pd.read_csv("gene_lengths.csv", index_col=0)
tpm_data = preprocessor.normalize(
    filtered_data,
    method='tpm',
    gene_lengths=gene_lengths['Length']
)

# Step 3: Log-transform the TPM values
log_tpm = preprocessor.normalize(
    tpm_data,
    method='log',
    pseudocount=1
)

# Step 4: Impute any missing values
final_data = preprocessor.impute(
    log_tpm,
    method='knn',
    n_neighbors=5
)

# Save results
final_data.to_csv("processed_data.csv")
```

### Processing Clinical Metadata

```python
# Load clinical data
clinical = pd.read_csv("clinical_data.csv", index_col=0)

# Initialize clinical processor
preprocessor = BioinformaticsPreprocessor(data_type='clinical')

# Process all in one step
processed_clinical = preprocessor.process_pipeline(
    clinical,
    steps=['filter', 'normalize', 'impute'],
    filter_kwargs={
        'method': 'missing',
        'threshold': 0.3  # Remove features with >30% missing values
    },
    normalize_kwargs={
        'method': 'auto',  # Automatically handle categorical and numerical
        'num_scaling': 'standard',
        'cat_encoding': 'onehot'
    },
    impute_kwargs={
        'method': 'auto'  # Use appropriate method for each data type
    }
)
```

### Processing Methylation Data

```python
# Load methylation beta values
methyl_data = pd.read_csv("methylation_beta.csv", index_col=0)

# Initialize methylation processor
preprocessor = BioinformaticsPreprocessor(data_type='methyl_seq')

# Filter by variance
filtered_methyl = preprocessor.filter_features(
    methyl_data,
    method='variance',
    min_variance=0.001
)

# Convert to M-values
m_values = preprocessor.normalize(
    filtered_methyl,
    method='m_value'
)

# Find most variable sites
top_variable = preprocessor.filter_features(
    m_values,
    method='variance',
    percentile=95  # Keep top 5% most variable sites
)
```

## Dependencies

- numpy
- pandas
- scipy
- scikit-learn
- logging

## Optional Dependencies

- rpy2 (for advanced methods that interface with R packages)

## Contact

For questions and feedback, please open an issue on GitHub or contact jiyang.jiang@gmail.com .
