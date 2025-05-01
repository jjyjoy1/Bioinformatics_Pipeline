# Bioinformatics ModelExplainer

A comprehensive toolkit for interpreting and explaining machine learning and deep learning models, specifically designed for bioinformatics applications but generally applicable to many domains.

## Features

- **Multi-model support**: Works with scikit-learn models, TensorFlow/Keras, and PyTorch
- **Comprehensive explanation methods**:
  - Feature importance extraction (native and permutation-based)
  - SHAP (SHapley Additive exPlanations) analysis
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Attention weights visualization for transformer/LSTM models
- **Rich visualizations**:
  - Feature importance plots
  - SHAP summary, dependence, and waterfall plots
  - LIME explanation plots
  - Attention heatmaps and aggregated attention visualization
- **Detailed reporting**: Generate comprehensive HTML reports with visualizations
- **Flexible integration**: Works with both tabular data and more complex structures

# Install required dependencies
pip install numpy pandas matplotlib seaborn scikit-learn joblib tqdm

# Install optional dependencies for additional functionality
pip install shap lime tensorflow torch
```

## Quick Start

```python
from bioinformatics_modelexplainer import BioinformaticsModelExplainer

# Initialize the explainer with your trained model
explainer = BioinformaticsModelExplainer(
    model=your_trained_model,
    task_type='classification'  # or 'regression'
)

# Compute feature importance
importance_df = explainer.compute_feature_importance(X_test)

# Generate SHAP values
shap_values = explainer.compute_shap_values(X_test)

# Create a comprehensive report
report = explainer.generate_explanation_report(
    X_test,
    output_format='html'
)
```

## Example Usage

See the [examples](examples/) directory for complete examples with different model types.

### Basic Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ModelExplainer import BioinformaticsModelExplainer

# Prepare your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train your model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Initialize the explainer
explainer = BioinformaticsModelExplainer(
    model=model, 
    task_type='classification'
)

# Compute feature importance
importance_df = explainer.compute_feature_importance(X_test)

# Plot feature importance
explainer.plot_feature_importance(n_features=10)

# Compute SHAP values
explainer.compute_shap_values(X_test, n_samples=100)

# Generate a comprehensive report
report = explainer.generate_explanation_report(
    X_test,
    output_format='html'
)

# Save report to file
with open("explanation_report.html", "w") as f:
    f.write(report)
```

## Explanation Methods

### Feature Importance

Extract feature importance from various model types using either:
- Native model feature importance (e.g., from Random Forests)
- Permutation importance
- SHAP-based importance

```python
# Native importance
importance_df = explainer.compute_feature_importance(X, method='native')

# Permutation importance
importance_df = explainer.compute_feature_importance(X, y, method='permutation')

# SHAP-based importance
explainer.compute_shap_values(X)
importance_df = explainer.compute_feature_importance(X, method='shap')
```

### SHAP Analysis

SHAP (SHapley Additive exPlanations) provides detailed analysis of feature contributions:

```python
# Compute SHAP values
shap_values = explainer.compute_shap_values(X)

# Create summary plot
explainer.plot_shap_summary()

# Create dependence plot to analyze feature interactions
explainer.plot_shap_dependence('feature_name')

# Create waterfall plot for a specific prediction
explainer.plot_shap_waterfall(instance_index=0)
```

### LIME Explanations

LIME (Local Interpretable Model-agnostic Explanations) gives instance-specific explanations:

```python
# Compute LIME explanations
lime_explanations = explainer.compute_lime_explanation(X, n_samples=5)

# Plot explanation for a specific instance
explainer.plot_lime_explanation(instance_index=0)
```

### Attention Weights Visualization

For transformer or LSTM models, visualize attention weights:

```python
# Compute attention weights
attention_weights = explainer.compute_attention_weights(X)

# Plot attention heatmap for a specific instance
explainer.plot_attention_heatmap(instance_index=0)

# Plot aggregated attention across all instances
explainer.plot_attention_aggregated(top_features=10)
```

## Comprehensive Reporting

Generate detailed reports combining multiple explanation methods:

```python
report = explainer.generate_explanation_report(
    X_test,
    y_test,
    output_format='html',  # or 'json', 'dict'
    compute_importance=True,
    compute_shap=True,
    compute_lime=True,
    compute_attention=True
)
```

## Model Compatibility

The toolkit automatically detects model types and uses appropriate explanation methods:

- **Traditional ML Models**: RandomForest, XGBoost, SVM, Linear/Logistic Regression, etc.
- **Deep Learning Models**: 
  - TensorFlow/Keras: Dense networks, CNNs, RNNs/LSTMs, Transformers
  - PyTorch: Various architectures

## Saving & Loading Explanations

```python
# Save explanations
explainer.save_explanations("explanations.json", format='json')

# Load explanations
loaded_explanations = BioinformaticsModelExplainer.load_explanations(
    "explanations.json", 
    format='json'
)
```

## Requirements

### Core Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tqdm
- joblib

### Optional Dependencies
- shap
- lime
- tensorflow (for deep learning models)
- torch (for PyTorch models)


## Acknowledgments

- SHAP: [https://github.com/slundberg/shap](https://github.com/slundberg/shap)
- LIME: [https://github.com/marcotcr/lime](https://github.com/marcotcr/lime)
