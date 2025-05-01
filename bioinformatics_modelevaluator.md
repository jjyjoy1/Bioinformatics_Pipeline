# Bioinformatics ModelEvaluator

A comprehensive framework for evaluating and optimizing machine learning and deep learning models in bioinformatics applications.

## 🧬 Overview

BioinformaticsModelEvaluator provides a robust toolkit for comprehensive assessment, optimization, and interpretation of ML/DL models specifically designed for bioinformatics tasks. It handles classification, regression, and clustering scenarios with specialized functionality for both traditional ML models and deep learning architectures.

## ✨ Features

- **Diverse Cross-Validation Strategies**: k-fold, stratified, repeated, leave-one-out, and shuffle-split CV
- **Hyperparameter Optimization**: 
  - Grid search
  - Random search
  - Bayesian optimization (via Optuna)
- **Comprehensive Evaluation Metrics**:
  - Classification: accuracy, precision, recall, F1, ROC AUC, confusion matrix
  - Regression: MSE, RMSE, MAE, R², explained variance
  - Clustering: silhouette score, Davies-Bouldin, Calinski-Harabasz
- **Statistical Validation**:
  - Bootstrap confidence intervals
  - Permutation feature importance
  - Model calibration
- **Advanced Model Interpretation**:
  - SHAP (SHapley Additive exPlanations) analysis
  - Permutation importance
  - Feature importance visualization
- **Learning Dynamics Analysis**:
  - Learning curves
  - Validation curves
- **Visualization Tools**:
  - Confusion matrices
  - ROC and Precision-Recall curves
  - Residual plots for regression
  - Feature importance charts
  - SHAP summary plots
- **Ensemble Methods Support**:
  - Voting ensembles
  - Stacking ensembles
- **Deep Learning Support**:
  - Custom handling for TensorFlow/Keras models
  - Deep learning-specific cross-validation strategies
  - Specialized hyperparameter optimization


## 📋 Requirements

- Python 3.7+
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- SciPy
- joblib

### Optional dependencies:
- Optuna (for Bayesian optimization)
- SHAP (for model interpretation)
- TensorFlow (for deep learning support)

## 🚀 Quick Start

```python
from bioinformatics_modelevaluator import BioinformaticsModelEvaluator
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Create a model
model = RandomForestClassifier(random_state=42)

# Initialize evaluator
evaluator = BioinformaticsModelEvaluator(task_type='classification')

# Perform cross-validation
cv_results = evaluator.cross_validate_model(model, X, y, cv_strategy='stratified', n_splits=5)

# Print results
print(f"Mean accuracy: {cv_results['summary']['test_accuracy']['mean']:.4f}")
```

## 📖 Usage Examples

### 1. Cross-Validation

```python
# Perform 5-fold stratified cross-validation
cv_results = evaluator.cross_validate_model(
    model, X, y, 
    cv_strategy='stratified', 
    n_splits=5,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
)

# Access the results
mean_accuracy = cv_results['summary']['test_accuracy']['mean']
std_accuracy = cv_results['summary']['test_accuracy']['std']
```

### 2. Hyperparameter Optimization

```python
# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Perform random search
opt_results = evaluator.optimize_hyperparameters(
    model, X, y,
    param_grid=param_grid,
    search_method='random',
    cv=5,
    n_iter=20
)

# Access the best parameters
best_params = evaluator.best_params_
best_score = evaluator.best_score_
```

### 3. Nested Cross-Validation

```python
# Perform nested cross-validation with hyperparameter tuning
nested_results = evaluator.nested_cross_validate(
    model, X, y, 
    param_grid=param_grid,
    outer_cv=5, 
    inner_cv=3,
    search_method='random'
)

# Access results
mean_score = nested_results['mean_score']
std_score = nested_results['std_score']
```

### 4. Model Evaluation

```python
# Evaluate on a test set
metrics = evaluator.evaluate_model(
    best_model, X, y, test_size=0.2
)

# Access evaluation metrics
accuracy = metrics['accuracy']
precision = metrics['precision']
recall = metrics['recall']
```

### 5. Feature Importance

```python
# Calculate permutation importance
importance_results = evaluator.permutation_importance(
    best_model, X, y, n_repeats=10
)

# Plot feature importance
fig = evaluator.plot_permutation_importance(n_features=10)
plt.savefig('feature_importance.png')
```

### 6. SHAP Analysis

```python
# Calculate SHAP values
shap_values = evaluator.compute_shap_values(best_model, X)

# Plot SHAP summary
shap_fig = evaluator.plot_shap_summary(shap_values, max_display=15)
plt.savefig('shap_summary.png')
```

### 7. Learning Curve Analysis

```python
# Analyze learning curves
lc_results = evaluator.learning_curve_analysis(
    best_model, X, y, cv=5
)

# Plot learning curve
fig = evaluator.plot_learning_curve(lc_results)
plt.savefig('learning_curve.png')
```

### 8. Working with Deep Learning Models

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Create a DL model
dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Initialize evaluator for classification
dl_evaluator = BioinformaticsModelEvaluator(task_type='classification')

# Optimize hyperparameters
dl_param_grid = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 150],
    'optimizer': ['adam', 'rmsprop']
}

dl_opt_results = dl_evaluator.optimize_hyperparameters(
    dl_model, X, y,
    param_grid=dl_param_grid,
    search_method='random',
    cv=3,
    n_iter=10,
    batch_size=32,
    epochs=50
)
```

## 📧 Contact

Jiyang Jiang   jiyang.jiang@gmail.com

