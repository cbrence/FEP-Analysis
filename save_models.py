import joblib
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# Create directory for saved models if it doesn't exist
saved_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
os.makedirs(saved_models_dir, exist_ok=True)

# Function to train and save a model
def train_and_save_model(model, model_name):
    # Example training data - replace with your actual dataset
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 8, 100)  # 8 classes
    
    # Fit the model
    model.fit(X, y)
    
    # Save the model
    save_path = os.path.join(saved_models_dir, f"{model_name}.joblib")
    joblib.dump(model, save_path)
    print(f"Model saved to: {save_path}")

# Create and save models
train_and_save_model(
    LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42),
    "Logistic_Regression"
)

train_and_save_model(
    DecisionTreeClassifier(random_state=42),
    "Decision_Tree"
)

train_and_save_model(
    GradientBoostingClassifier(random_state=42),
    "Gradient_Boosting"
)

train_and_save_model(
    MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
    "Neural_Network"
)

# You can also define a custom ensemble model
class HighRiskEnsemble:
    def __init__(self):
        self.base_models = [
            GradientBoostingClassifier(random_state=42),
            LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
        ]
        self.weights = [0.7, 0.3]
    
    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
        return self
    
    def predict(self, X):
        # Simple weighted voting
        predictions = np.array([model.predict(X) for model in self.base_models])
        weighted_preds = np.average(predictions, axis=0, weights=self.weights)
        return np.round(weighted_preds).astype(int)
    
    def predict_proba(self, X):
        # Weighted average of probabilities
        probas = np.array([model.predict_proba(X) for model in self.base_models])
        return np.average(probas, axis=0, weights=self.weights)

# Train and save the ensemble model
train_and_save_model(
    HighRiskEnsemble(),
    "High_Risk_Ensemble"
)

print("All models saved successfully!")