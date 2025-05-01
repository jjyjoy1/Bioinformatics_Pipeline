import pandas as pd
import numpy as np
from bioinformatics_biointerpreter import BiologicalInterpreter

# Initialize the interpreter
bio_interp = BiologicalInterpreter(organism="human", data_dir="./data")

# Load some example differential expression data
# In a real scenario, this would come from your RNA-seq or proteomics analysis
gene_ids = ["3101", "3098", "3099", "4967", "4968", "1737", "1738", "1432", "5594", "5595", "6416", "2821"]
log2fc = [2.5, 1.8, 2.1, -1.5, -1.2, -0.8, -1.9, 0.5, 1.2, 0.3, 1.6, 2.2]
pvalues = [0.001, 0.005, 0.002, 0.01, 0.03, 0.04, 0.005, 0.2, 0.01, 0.3, 0.007, 0.001]

# Create a dataframe
de_results = pd.DataFrame({
    'gene_id': gene_ids,
    'log2fc': log2fc,
    'pvalue': pvalues
})

# Filter for significant genes (typical RNA-seq analysis threshold)
significant_genes = de_results[de_results['pvalue'] < 0.05]['gene_id'].tolist()

# Run Over-Representation Analysis (ORA) on KEGG pathways
kegg_results = bio_interp.run_enrichment_analysis(
    gene_list=significant_genes,
    source="KEGG",
    method="ora",
    pvalue_cutoff=0.05
)

# Display results
print("KEGG Pathway Enrichment Results:")
print(kegg_results)

# Visualize the enrichment results
bio_interp.visualize_enrichment(kegg_results, n_terms=10, plot_type='barplot')

# Generate a report
report_path = bio_interp.generate_pathway_report(
    results=kegg_results,
    source="KEGG",
    output_dir="./reports",
    include_genes=True,
    include_plots=True
)

print(f"Report generated at: {report_path}")

# Run Gene Set Enrichment Analysis (GSEA)
# Create gene scores dictionary (log2fc values)
gene_scores = dict(zip(de_results['gene_id'], de_results['log2fc']))

gsea_results = bio_interp.run_gsea(
    gene_scores=gene_scores,
    source="REACTOME",
    permutations=100,  # Use more permutations (e.g. 1000) for publication-quality results
    min_set_size=3,  # Reduced for this example
    max_set_size=500,
    pvalue_cutoff=0.05
)

# Display GSEA results
print("\nGSEA Results for Reactome Pathways:")
print(gsea_results)

# Create a network visualization
bio_interp.create_network_visualization(
    results=kegg_results,
    n_terms=5,
    include_genes=True,
    output_path="./reports/network_viz.png"
)

# Export results for downstream use
bio_interp.export_results(
    results=kegg_results,
    output_path="./exports/kegg_enrichment.csv",
    format="csv"
)


