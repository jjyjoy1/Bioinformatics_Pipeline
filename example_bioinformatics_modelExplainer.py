import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Import the explainer class
from ModelExplainer import BioinformaticsModelExplainer

# Set up logger
logger = logging.getLogger("ModelExplainerExample")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load example dataset
logger.info("Loading breast cancer dataset")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a model
logger.info("Training Random Forest model")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
logger.info(f"Model accuracy: {model.score(X_test_scaled, y_test):.4f}")

# Initialize the explainer
logger.info("Initializing explainer")
explainer = BioinformaticsModelExplainer(
    model=model, 
    task_type='classification',
    logger=logger
)

# 1. Compute feature importance
logger.info("Computing feature importance")
importance_df = explainer.compute_feature_importance(
    X_test_scaled, 
    method='native'
)
print("\nTop 10 features by importance:")
print(importance_df.head(10))

# 2. Plot feature importance
logger.info("Plotting feature importance")
fig = explainer.plot_feature_importance(n_features=10)
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# 3. Compute SHAP values (if available)
try:
    logger.info("Computing SHAP values")
    shap_values = explainer.compute_shap_values(
        X_test_scaled, 
        n_samples=100,
        background_samples=50
    )
    
    # 4. Create SHAP summary plot
    logger.info("Creating SHAP summary plot")
    fig = explainer.plot_shap_summary(plot_type='bar')
    plt.tight_layout()
    plt.savefig("shap_summary.png")
    plt.close()
    
    # 5. Create SHAP waterfall plot for a specific instance
    logger.info("Creating SHAP waterfall plot")
    fig = explainer.plot_shap_waterfall(instance_index=0)
    plt.tight_layout()
    plt.savefig("shap_waterfall.png")
    plt.close()
except ImportError:
    logger.warning("SHAP not installed, skipping SHAP analysis")

# 6. Compute LIME explanations (if available)
try:
    logger.info("Computing LIME explanations")
    lime_explanations = explainer.compute_lime_explanation(
        X_test_scaled, 
        n_samples=5
    )
    
    # 7. Plot LIME explanation for a specific instance
    logger.info("Plotting LIME explanation")
    fig = explainer.plot_lime_explanation(instance_index=0)
    plt.tight_layout()
    plt.savefig("lime_explanation.png")
    plt.close()
except ImportError:
    logger.warning("LIME not installed, skipping LIME analysis")

# 8. Generate comprehensive explanation report
logger.info("Generating explanation report")
report = explainer.generate_explanation_report(
    X_test_scaled,
    y_test,
    output_format='html',
    n_samples=5,
    compute_importance=True,
    compute_shap=True,
    compute_lime=True
)

# Save report to file
with open("model_explanation_report.html", "w") as f:
    f.write(report)

logger.info("Explanation report saved to model_explanation_report.html")

# 9. Save explanations
logger.info("Saving explanations")
explainer.save_explanations("explanations.json", format='json')

# Example on how to load saved explanations
loaded_explanations = BioinformaticsModelExplainer.load_explanations(
    "explanations.json", 
    format='json'
)

logger.info("Done!")




