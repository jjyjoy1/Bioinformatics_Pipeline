import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from bioinformatics_classification_regression import BioinformaticsSupervisedLearner
import matplotlib.pyplot as plt

# Example 1: Classification with Random Forest
# ------------------------------------

# Load some example data (replace with your bioinformatics dataset)
# For example: gene expression data for disease classification
data = pd.read_csv('gene_expression_data.csv')
X = data.drop('disease_status', axis=1)  # Features: gene expression values
y = data['disease_status']               # Target: disease status (e.g., 'healthy', 'diseased')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocess data - standardize features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train), 
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# Create and train classifier
classifier = BioinformaticsSupervisedLearner(
    task_type='classification',
    model_type='random_forest'
)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Train with hyperparameter tuning
classifier.fit(
    X_train_scaled, 
    y_train,
    hyperparameter_tuning=True,
    param_grid=param_grid,
    cv=5
)

# Evaluate on test set
metrics = classifier.evaluate(X_test_scaled, y_test)
print("Classification Results:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1']:.4f}")
print(f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")

# Visualize results
fig = classifier.plot_confusion_matrix()
plt.savefig('confusion_matrix.png')
plt.close()

fig = classifier.plot_feature_importances(n_features=15)
plt.savefig('feature_importances.png')
plt.close()

# Make predictions on new data
new_samples = X_test_scaled.iloc[:5]  # Just as an example
predictions = classifier.predict(new_samples)
print("\nSample Predictions:")
print(predictions)

# Example 2: Regression with Neural Network (MLP)
# ------------------------------------

# Load regression dataset (replace with your bioinformatics dataset)
# For example: predicting protein binding affinity from sequence features
regression_data = pd.read_csv('protein_binding_data.csv')
X_reg = regression_data.drop('binding_affinity', axis=1)  # Features
y_reg = regression_data['binding_affinity']              # Target: continuous value

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Preprocess data
X_train_reg_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_reg),
    columns=X_train_reg.columns
)
X_test_reg_scaled = pd.DataFrame(
    scaler.transform(X_test_reg),
    columns=X_test_reg.columns
)

# Create and train regressor
regressor = BioinformaticsSupervisedLearner(
    task_type='regression',
    model_type='mlp'
)

# Train the model with custom parameters
regressor.fit(
    X_train_reg_scaled,
    y_train_reg,
    hidden_layers=[128, 64, 32],
    activation='relu',
    dropout_rate=0.3,
    batch_norm=True,
    batch_size=32,
    epochs=100,
    early_stopping=True,
    patience=10,
    validation_split=0.2
)

# Evaluate on test set
reg_metrics = regressor.evaluate(X_test_reg_scaled, y_test_reg)
print("\nRegression Results:")
print(f"RMSE: {reg_metrics['rmse']:.4f}")
print(f"MAE: {reg_metrics['mae']:.4f}")
print(f"R²: {reg_metrics['r2']:.4f}")
print(f"Pearson correlation: {reg_metrics['pearson_r']:.4f} (p={reg_metrics['pearson_p']:.4f})")

# Visualize results
fig = regressor.plot_regression_results()
plt.savefig('regression_results.png')
plt.close()

if hasattr(regressor, 'history_'):
    fig = regressor.plot_learning_curve()
    plt.savefig('learning_curve.png')
    plt.close()

# Save trained models
classifier.save_model('models/gene_expression_classifier')
regressor.save_model('models/protein_binding_regressor')

# Later, load models
loaded_classifier = BioinformaticsSupervisedLearner.load_model('models/gene_expression_classifier')
loaded_regressor = BioinformaticsSupervisedLearner.load_model('models/protein_binding_regressor')


