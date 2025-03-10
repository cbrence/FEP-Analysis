from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from .base import BaseFEPModel

class DecisionTreeFEP(BaseFEPModel):
    """Decision Tree model for FEP prediction with cost-sensitive learning."""
    
    def __init__(self, class_weight='clinical', random_state=42, param_grid=None):
        super().__init__(class_weight=class_weight, random_state=random_state)
        
        # Default parameter grid if none provided
        self.param_grid = param_grid or {
            "max_depth": range(2, 8),
            "min_samples_leaf": range(5, 55, 5),
            "min_samples_split": range(5, 110, 5)
        }
        
    def _fit_model(self, X, y, sample_weight=None):
        """Implementation of model fitting with grid search."""
        base_estimator = DecisionTreeClassifier(random_state=self.random_state)
        
        # Configure grid search
        self.grid_search = GridSearchCV(
            base_estimator,
            param_grid=self.param_grid,
            scoring='roc_auc_ovr_weighted',
            n_jobs=-1,
            cv=5
        )
        
        # Fit the model
        self.grid_search.fit(X, y, sample_weight=sample_weight)
        
        # Store best model
        self.model = self.grid_search.best_estimator_
        
    def _predict_proba_implementation(self, X):
        """Implementation of probability prediction."""
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        """Extract feature importance from model."""
        if not hasattr(self, 'model') or not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model not fitted or does not support feature importance")
            
        return self.model.feature_importances_
        
    def get_best_params(self):
        """Get the best parameters found by grid search."""
        if not hasattr(self, 'grid_search'):
            raise AttributeError("Model not fitted with grid search")
            
        return self.grid_search.best_params_