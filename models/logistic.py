from sklearn.linear_model import LogisticRegressionCV
from .base import BaseFEPModel

class LogisticRegressionFEP(BaseFEPModel):
    """
    Logistic Regression model for FEP prediction with cost-sensitive learning.
    """
    
    def __init__(self, class_weight='clinical', cv=10, solver='saga', penalty='l1', random_state=42):
        super().__init__(class_weight=class_weight, random_state=random_state)
        self.cv = cv
        self.solver = solver
        self.penalty = penalty
        
    def _fit_model(self, X, y, sample_weight=None):
        """Implementation of model fitting."""
        self.model = LogisticRegressionCV(
            cv=self.cv,
            multi_class="ovr",
            solver=self.solver,
            penalty=self.penalty,
            random_state=self.random_state
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        
    def _predict_proba_implementation(self, X):
        """Implementation of probability prediction."""
        return self.model.predict_proba(X)
        
    def get_feature_importance(self):
        """Extract feature importance from model."""
        if not hasattr(self, 'model') or not hasattr(self.model, 'coef_'):
            raise AttributeError("Model not fitted or does not support feature importance")
            
        # For multi-class, average the absolute coefficients across classes
        if len(self.model.coef_.shape) > 1:
            importance = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            importance = np.abs(self.model.coef_)
            
        return importance