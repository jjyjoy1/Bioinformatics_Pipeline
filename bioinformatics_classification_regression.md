# Bioinformatics Classification & Regression 

A comprehensive Python library for supervised learning tasks in bioinformatics, supporting classification and regression with both traditional machine learning and deep learning approaches.

## Overview

The `BioinformaticsSupervisedLearner` is designed to simplify the application of machine learning techniques to bioinformatics data. It provides a unified interface to various algorithms while handling common tasks such as:

- Model training and evaluation
- Hyperparameter optimization
- Cross-validation
- Performance metrics calculation
- Model visualization
- Feature importance analysis
- Model persistence

## Features

### Supported Tasks
- **Classification**: Predict categorical outcomes, phenotypes, or conditions
- **Regression**: Predict continuous biological variables or measurements

### Supported Models
- **Traditional ML**:
  - Logistic/Linear Regression
  - Random Forest
  - Support Vector Machines (SVM)
  - Gradient Boosting
  - XGBoost

- **Deep Learning**:
  - Multi-layer Perceptron (MLP)
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN/LSTM)
  - Transformer-based models

### Additional Capabilities
- Automatic hyperparameter tuning
- Cross-validation for robust performance estimation
- Rich visualization tools (confusion matrices, ROC curves, feature importance, etc.)
- Comprehensive metrics for model evaluation
- Easy model saving and loading


### Dependencies

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- joblib

### Optional Dependencies

- tensorflow (for deep learning models)
- xgboost (for XGBoost models)

## Usage Examples

### Classification Example

```python
from bioinformatics_classification_regression import BioinformaticsSupervisedLearner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load data (gene expression data for disease classification)
data = pd.read_csv('gene_expression_data.csv')
X = data.drop('disease_status', axis=1)
y = data['disease_status']

# Split and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Create and train classifier
classifier = BioinformaticsSupervisedLearner(
    task_type='classification',
    model_type='random_forest'
)

# Train with hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20]
}

classifier.fit(
    X_train_scaled, 
    y_train,
    hyperparameter_tuning=True,
    param_grid=param_grid,
    cv=5
)

# Evaluate
metrics = classifier.evaluate(X_test_scaled, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")

# Visualize
classifier.plot_confusion_matrix()
classifier.plot_feature_importances(n_features=15)

# Save model
classifier.save_model('models/disease_classifier')
```

### Regression Example

```python
# Create a regressor for protein binding affinity prediction
regressor = BioinformaticsSupervisedLearner(
    task_type='regression',
    model_type='mlp'
)

# Train with custom neural network parameters
regressor.fit(
    X_train_scaled,
    y_train,
    hidden_layers=[128, 64, 32],
    activation='relu',
    dropout_rate=0.3,
    batch_norm=True,
    early_stopping=True
)

# Evaluate
metrics = regressor.evaluate(X_test_scaled, y_test)
print(f"R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.4f}")

# Visualize
regressor.plot_regression_results()
regressor.plot_learning_curve()
```

## API Reference

### Initialization

```python
BioinformaticsSupervisedLearner(
    task_type='classification',   # 'classification' or 'regression'
    model_type='random_forest',   # Model type to use
    random_state=42,              # For reproducibility
    logger=None                   # Optional custom logger
)
```

### Main Methods

- **fit(X, y, **kwargs)**: Train the model
- **predict(X)**: Make predictions
- **predict_proba(X)**: Get probability predictions (classification only)
- **evaluate(X, y)**: Evaluate model performance
- **cross_validate(X, y, cv=5)**: Perform cross-validation
- **save_model(filepath)**: Save the trained model
- **load_model(filepath)**: Load a trained model (class method)

### Visualization Methods

- **plot_feature_importances(n_features=20)**: Plot top feature importances
- **plot_learning_curve()**: Plot training and validation metrics (deep learning)
- **plot_confusion_matrix()**: Plot confusion matrix (classification)
- **plot_roc_curve()**: Plot ROC curve (classification)
- **plot_regression_results()**: Plot actual vs predicted values (regression)
- **plot_permutation_importance(X, y)**: Plot permutation feature importance


