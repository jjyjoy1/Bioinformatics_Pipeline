## ML‑Driven Bioinformatics Toolkit (Overview)

We have developed a versatile, reusable **Python module** that streamlines matrix‑based biomedical analyses (RNA‑seq, ChIP‑seq, metagenomics, eQTL/WGAS, etc.). The toolkit wraps each common computational task in a clean, object‑oriented API so that teams can drop the classes into any workflow, iterate quickly, and reproduce results easily.

* **Data normalization & QC** – class `Normalizer` supports DESeq2‑style size‑factor scaling, TMM, CLR, log/variance‑stabilizing transforms, and custom functions.
* **Feature selection** – class `FeatureSelector` offers statistical tests, variance filters, L1/ElasticNet, tree‑based importance, RFE, and SHAP‑driven pruning.
* **Dimensionality reduction** – class `Reducer` unifies PCA, ICA, t‑SNE, UMAP, autoencoders, and VAEs with identical `.fit_transform()` signatures.
* **Clustering & network analysis** – class `Clusterer` implements K‑means, hierarchical, DBSCAN, Gaussian Mixture, spectral, and WGCNA wrappers.
* **Supervised learning** – class `Classifier`/`Regressor` abstracts scikit‑learn estimators, XGBoost/LightGBM, and custom PyTorch/Keras models in one API.
* **Model evaluation & tuning** – class `Evaluator` automates CV splits, Optuna/Bayesian hyper‑opt, metric dashboards, and permutation importance.
* **Explainable AI** – helpers for SHAP, LIME, attention heat‑maps, feature attribution plots.
* **Biological interpretation** – built‑ins for GO/KEGG/Reactome enrichment, GSEA, and pathway plotting.

### Generalized Pipeline (Mermaid Diagram)
```mermaid
flowchart TD
    A[Raw Data\n(samples × features)] --> B[Normalization & Preprocessing]
    B --> C[Feature Selection]
    C --> D[Dimensionality Reduction]
    D --> E[Clustering]
    D --> F[Supervised Analysis]
    E --> G[Unsupervised Interpretation]
    F --> H[Model Evaluation & Optimization]
    H --> I[Explainable AI]
    I --> J[Functional & Pathway Enrichment]
    J --> K[Biological Insight & Hypothesis]
```

