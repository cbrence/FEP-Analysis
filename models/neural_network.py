from sklearn.neural_network import MLPClassifier
from .base import BaseFEPModel
import numpy as np

class NeuralNetworkFEP(BaseFEPModel):
    """Neural Network model for FEP prediction with cost-sensitive learning."""
    
    def __init__(self, hidden_layer_sizes=(6, 6), activation='relu', learning_rate=0.05, 
                 max_iter=250, class_weight='clinical', random_state=42):
        super().__init__(class_weight=class_weight, random_state=random_state)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
    def _fit_model(self, X, y, sample_weight=None):
        """Implementation of model fitting."""
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        
    def _predict_proba_implementation(self, X):
        """Implementation of probability prediction."""
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        """
        Extract feature importance from neural network.
        For neural networks, this is a rough approximation using first layer weights.
        """
        if not hasattr(self, 'model') or not hasattr(self.model, 'coefs_'):
            raise AttributeError("Model not fitted or does not have accessible weights")
            
        # Use magnitude of first layer weights as rough importance metric
        first_layer_weights = self.model.coefs_[0]
        return np.mean(np.abs(first_layer_weights), axis=1)