# Biological Interpreter

A Python toolkit for biological interpretation of feature selection results, including pathway analysis and functional enrichment.

## Features

- 🧬 **Pathway Analysis**: Perform Over-Representation Analysis (ORA) and Gene Set Enrichment Analysis (GSEA)
- 🔍 **Multiple Data Sources**: Integrate with KEGG, GO, Reactome, and Disease Ontology
- 📊 **Visualization**: Create publication-ready visualizations of enrichment results
- 🌐 **Network Analysis**: Generate interactive network visualizations of pathway relationships
- 📝 **Reporting**: Generate comprehensive HTML reports for your analysis
- 📤 **Data Export**: Export results in various formats for downstream analysis

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- requests
- networkx (optional, for network visualization)

## Quick Start

```python
from bioinformatics_biointerpreter import BiologicalInterpreter

# Initialize the interpreter
bio_interp = BiologicalInterpreter(organism="human")

# Load some example genes
significant_genes = ["3101", "3098", "3099", "4967", "4968", "1737", "1738"]

# Run Over-Representation Analysis (ORA) on KEGG pathways
kegg_results = bio_interp.run_enrichment_analysis(
    gene_list=significant_genes,
    source="KEGG",
    method="ora",
    pvalue_cutoff=0.05
)

# Visualize the results
bio_interp.visualize_enrichment(kegg_results)

# Generate a comprehensive report
report_path = bio_interp.generate_pathway_report(
    results=kegg_results,
    source="KEGG",
    output_dir="./reports"
)
```

## Documentation

### BiologicalInterpreter Class

The main class for biological interpretation of selected features.

```python
BiologicalInterpreter(organism="human", data_dir="./data")
```

**Parameters:**
- `organism`: Organism for pathway analysis ('human', 'mouse', etc.)
- `data_dir`: Directory to store downloaded annotation files

### Key Methods

#### Loading Gene Sets

```python
load_gene_set(source, category=None)
```
- `source`: Source of gene sets ('KEGG', 'GO', 'REACTOME', 'DO')
- `category`: For GO, specify category ('BP', 'MF', 'CC')

#### Running Enrichment Analysis

```python
run_enrichment_analysis(
    gene_list, 
    source, 
    category=None,
    method='ora', 
    background_size=None,
    pvalue_cutoff=0.05
)
```

```python
run_gsea(
    gene_scores, 
    source, 
    category=None,
    permutations=1000,
    min_set_size=15,
    max_set_size=500,
    pvalue_cutoff=0.05
)
```

#### Visualization

```python
visualize_enrichment(
    results, 
    n_terms=20,
    save_path=None,
    plot_type='barplot'  # 'barplot', 'bubble', or 'heatmap'
)
```

```python
create_network_visualization(
    results,
    n_terms=10,
    include_genes=True,
    min_edge_weight=0.3,
    output_path=None
)
```

#### Reporting and Export

```python
generate_pathway_report(
    results,
    source,
    output_dir="./reports",
    include_genes=True,
    include_plots=True
)
```

```python
export_results(
    results, 
    output_path,
    format='csv'  # 'csv', 'tsv', 'excel', 'json'
)
```

## Example Use Cases

### Analyzing RNA-seq Differential Expression Results

```python
# Load your differential expression results
de_results = pd.read_csv("differential_expression.csv")

# Extract significant genes
significant_genes = de_results[de_results['padj'] < 0.05]['gene_id'].tolist()

# Run pathway analysis
bio_interp = BiologicalInterpreter()
go_results = bio_interp.run_enrichment_analysis(
    gene_list=significant_genes,
    source="GO",
    category="BP",  # Biological Process
    method="ora"
)

# Generate a report
bio_interp.generate_pathway_report(go_results, "GO", include_plots=True)
```

### Running GSEA on Proteomics Data

```python
# Load proteomics results with fold changes
proteomic_data = pd.read_csv("proteomics_data.csv")

# Create a dictionary of gene scores (e.g., log2 fold changes)
gene_scores = dict(zip(proteomic_data['protein_id'], proteomic_data['log2fc']))

# Run GSEA
bio_interp = BiologicalInterpreter(organism="human")
gsea_results = bio_interp.run_gsea(
    gene_scores=gene_scores,
    source="REACTOME",
    permutations=1000
)

# Visualize top enriched pathways
bio_interp.visualize_enrichment(gsea_results, n_terms=15, plot_type='bubble')
```


## Contact

Jiyang Jiang - jiyang.jiang@gmail.com

Project Link: [https://github.com/yourusername/BiologicalInterpreter](https://github.com/yourusername/BiologicalInterpreter)
