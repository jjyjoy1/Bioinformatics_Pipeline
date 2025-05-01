import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

# Import the explainer class
from ModelExplainer import BioinformaticsModelExplainer

# Set up logger
logger = logging.getLogger("DeepLearningExplainerExample")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Generate synthetic time series data for gene expression
def generate_synthetic_data(n_samples=1000, n_features=20, n_timesteps=10):
    """Generate synthetic time series data for demonstration"""
    logger.info(f"Generating synthetic data with {n_samples} samples, {n_features} features, {n_timesteps} timesteps")
    
    # Generate feature matrix
    X = np.random.randn(n_samples, n_timesteps, n_features)
    
    # Generate target: influenced by specific patterns in the data
    y = np.zeros(n_samples)
    
    # Rule 1: If feature 0 increases over time, class 1
    feature_0_trend = np.diff(X[:, :, 0], axis=1).mean(axis=1)
    y[feature_0_trend > 0.2] = 1
    
    # Rule 2: If feature 5 has high variance, class 1
    feature_5_var = np.var(X[:, :, 5], axis=1)
    y[feature_5_var > 1.2] = 1
    
    # Rule 3: If specific pattern in feature 10 and 11, class 1
    pattern = (X[:, -1, 10] > 0.5) & (X[:, -1, 11] < -0.3)
    y[pattern] = 1
    
    # Convert to binary classification
    y = y.astype(int)
    
    # Create feature names
    feature_names = [f'gene_{i}' for i in range(n_features)]
    
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_names

# Generate data
X, y, feature_names = generate_synthetic_data(n_samples=1000, n_features=20, n_timesteps=10)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
logger.info("Building LSTM model")
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, name='lstm_1'),
    Dropout(0.2),
    LSTM(32, name='lstm_2'),
    Dense(16, activation='relu', name='dense_1'),
    Dropout(0.2),
    Dense(1, activation='sigmoid', name='output')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Train model
logger.info("Training LSTM model")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    verbose=1
)

# Evaluate model
logger.info("Evaluating model")
loss, accuracy = model.evaluate(X_test, y_test)
logger.info(f"Test accuracy: {accuracy:.4f}")

# Reshape test data for the explainer
# The explainer expects 2D data, so we'll flatten the time dimension
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create flattened feature names
flat_feature_names = []
for i in range(X.shape[1]):  # For each timestep
    for j in range(X.shape[2]):  # For each feature
        flat_feature_names.append(f'{feature_names[j]}_t{i}')

# Initialize the explainer
logger.info("Initializing explainer")
explainer = BioinformaticsModelExplainer(
    model=model, 
    task_type='classification',
    logger=logger
)

# Compute permutation importance
logger.info("Computing permutation importance")
try:
    importance_df = explainer.compute_feature_importance(
        X_test_flat, y_test,
        method='permutation',
        n_repeats=5
    )
    print("\nTop 10 features by importance:")
    print(importance_df.head(10))
    
    # Plot feature importance
    fig = explainer.plot_feature_importance(n_features=15)
    plt.tight_layout()
    plt.savefig("lstm_feature_importance.png")
    plt.close()
except Exception as e:
    logger.error(f"Error computing permutation importance: {str(e)}")

# Compute SHAP values (if available)
try:
    logger.info("Computing SHAP values")
    shap_values = explainer.compute_shap_values(
        X_test_flat,
        n_samples=50,  # Using a small sample for demonstration
        background_samples=30,
        explainer_type='gradient'  # Using gradient explainer for LSTM
    )
    
    # Create SHAP summary plot
    logger.info("Creating SHAP summary plot")
    fig = explainer.plot_shap_summary(plot_type='bar')
    plt.tight_layout()
    plt.savefig("lstm_shap_summary.png")
    plt.close()
    
    # Create SHAP dependence plot for an important feature
    if len(importance_df) > 0:
        important_feature = importance_df.iloc[0]['feature']
        logger.info(f"Creating SHAP dependence plot for {important_feature}")
        fig = explainer.plot_shap_dependence(important_feature)
        plt.tight_layout()
        plt.savefig("lstm_shap_dependence.png")
        plt.close()
except ImportError:
    logger.warning("SHAP not installed, skipping SHAP analysis")
except Exception as e:
    logger.error(f"Error in SHAP analysis: {str(e)}")

# Compute LIME explanations (if available)
try:
    logger.info("Computing LIME explanations")
    lime_explanations = explainer.compute_lime_explanation(
        X_test_flat, 
        n_samples=5
    )
    
    # Plot LIME explanation for a specific instance
    logger.info("Plotting LIME explanation")
    fig = explainer.plot_lime_explanation(instance_index=0)
    plt.tight_layout()
    plt.savefig("lstm_lime_explanation.png")
    plt.close()
except ImportError:
    logger.warning("LIME not installed, skipping LIME analysis")
except Exception as e:
    logger.error(f"Error in LIME analysis: {str(e)}")

# Try to extract attention weights (if model has attention layers)
try:
    logger.info("Attempting to extract attention weights")
    attention_weights = explainer.compute_attention_weights(
        X_test_flat,
        n_samples=10
    )
    
    # Plot attention heatmap
    fig = explainer.plot_attention_heatmap(instance_index=0)
    plt.tight_layout()
    plt.savefig("lstm_attention_heatmap.png")
    plt.close()
except Exception as e:
    logger.warning(f"Could not extract attention weights: {str(e)}")
    logger.info("This is expected for simple LSTM models without explicit attention layers")

# Generate comprehensive explanation report
logger.info("Generating explanation report")
report = explainer.generate_explanation_report(
    X_test_flat,
    y_test,
    output_format='html',
    n_samples=5,
    compute_importance=True,
    compute_shap=True,
    compute_lime=True,
    compute_attention=False  # Set to False since our simple LSTM doesn't have attention
)

# Save report to file
with open("lstm_explanation_report.html", "w") as f:
    f.write(report)

logger.info("Explanation report saved to lstm_explanation_report.html")

# Save explanations
logger.info("Saving explanations")
explainer.save_explanations("lstm_explanations.json", format='json')

logger.info("Done!")

