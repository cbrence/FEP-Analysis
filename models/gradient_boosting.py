from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from .base import BaseFEPModel

class GradientBoostingFEP(BaseFEPModel):
    """Gradient Boosting model for FEP prediction with cost-sensitive learning."""
    
    def __init__(self, class_weight='clinical', random_state=42, param_dist=None, n_iter=20):
        super().__init__(class_weight=class_weight, random_state=random_state)
        
        # Default parameter distribution if none provided
        self.param_dist = param_dist or {
            "learning_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "max_depth": range(3, 8),
            "n_estimators": range(100, 150, 10),
            "subsample": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "min_samples_leaf": range(1, 10)
        }
        
        self.n_iter = n_iter
        
    def _fit_model(self, X, y, sample_weight=None):
        """Implementation of model fitting with randomized search."""
        base_estimator = GradientBoostingClassifier(random_state=self.random_state)
        
        # Configure randomized search
        self.random_search = RandomizedSearchCV(
            base_estimator,
            param_distributions=self.param_dist,
            n_iter=self.n_iter,
            scoring='roc_auc_ovr_weighted',
            n_jobs=-1,
            cv=5,
            random_state=self.random_state
        )
        
        # Fit the model
        self.random_search.fit(X, y, sample_weight=sample_weight)
        
        # Store best model
        self.model = self.random_search.best_estimator_
        
    def _predict_proba_implementation(self, X):
        """Implementation of probability prediction."""
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        """Extract feature importance from model."""
        if not hasattr(self, 'model') or not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model not fitted or does not support feature importance")
            
        return self.model.feature_importances_
        
    def get_best_params(self):
        """Get the best parameters found by randomized search."""
        if not hasattr(self, 'random_search'):
            raise AttributeError("Model not fitted with randomized search")
            
        return self.random_search.best_params_