import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import the evaluator
from bioinformatics_modelevaluator import BioinformaticsModelEvaluator

# Load a sample dataset (breast cancer)
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a model
rf_model = RandomForestClassifier(random_state=42)

# Initialize the evaluator
evaluator = BioinformaticsModelEvaluator(task_type='classification', random_state=42)

# 1. Perform cross-validation
cv_results = evaluator.cross_validate_model(
    rf_model, X_train_scaled, y_train, 
    cv_strategy='stratified', 
    n_splits=5
)

print("Cross-validation results:")
print(f"Mean accuracy: {cv_results['summary']['test_accuracy']['mean']:.4f} ± {cv_results['summary']['test_accuracy']['std']:.4f}")

# 2. Hyperparameter optimization
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

opt_results = evaluator.optimize_hyperparameters(
    rf_model, X_train_scaled, y_train,
    param_grid=param_grid,
    search_method='random',
    cv=5,
    n_iter=20
)

print("\nBest hyperparameters:")
for param, value in evaluator.best_params_.items():
    print(f"{param}: {value}")

print(f"Best score: {evaluator.best_score_:.4f}")

# 3. Evaluate the optimized model
best_model = opt_results.get('best_estimator')
eval_metrics = evaluator.evaluate_model(
    best_model, X_train_scaled, y_train, test_size=0.2
)

print("\nEvaluation metrics on validation set:")
for metric, value in eval_metrics.items():
    if metric != 'confusion_matrix':
        print(f"{metric}: {value:.4f}")

# 4. Feature importance analysis
importance_results = evaluator.permutation_importance(
    best_model, X_test_scaled, y_test, n_repeats=10
)

print("\nTop 5 features by importance:")
for i, row in importance_results['importance_df'].head(5).iterrows():
    print(f"{row['feature']}: {row['importance_mean']:.4f} ± {row['importance_std']:.4f}")

# 5. Perform learning curve analysis
lc_results = evaluator.learning_curve_analysis(
    best_model, X_train_scaled, y_train, cv=5
)

# 6. Plot the learning curve
fig = evaluator.plot_learning_curve(lc_results, title='Learning Curve - Random Forest')
plt.savefig('learning_curve.png')

# 7. Plot the confusion matrix
cm_fig = evaluator.plot_confusion_matrix()
plt.savefig('confusion_matrix.png')

# 8. Plot feature importance
importance_fig = evaluator.plot_permutation_importance(n_features=10)
plt.savefig('feature_importance.png')

# 9. Save the evaluation results
evaluator.save_results('model_evaluation_results.pkl', 
                      metadata={'model': 'RandomForest', 'dataset': 'Breast Cancer'})

print("\nEvaluation complete. Results saved to model_evaluation_results.pkl")

