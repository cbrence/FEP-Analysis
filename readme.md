# FEP Outcome Prediction with Cost-Sensitive Metrics

This project implements machine learning models to predict remission outcomes in First Episode Psychosis (FEP) patients, with a specific focus on preventing relapse through cost-sensitive learning approaches.

This project is based on an older project done in 2021 as part of Prof. Andrew Treadway's CIS 9660 Data Mining course at CUNY Baruch. Additional enhancements have been added such as modularization and dashboards for use by non-technical users. 

## Project Overview

Based on the results our models, First Episode Psychosis outcomes are typically categorized into several patterns based on remission at different time points:

- **Class 0**: No remission at 6 months, No remission at 12 months, Poor treatment adherence (Highest risk)
- **Class 1**: No remission at 6 months, No remission at 12 months, Moderate treatment adherence (Very high risk)
- **Class 2**: Remission at 6 months, No remission at 12 months - Early Relapse with significant functional decline (High risk)
- **Class 3**: No remission at 6 months, Remission at 12 months, Poor treatment adherence (Moderate-high risk)
- **Class 4**: Remission at 6 months, No remission at 12 months, Maintained social functioning (Moderate risk)
- **Class 5**: No remission at 6 months, Remission at 12 months, Good treatment adherence (Moderate-low risk)
- **Class 6**: Remission at 6 months, Remission at 12 months with some residual symptoms (Low risk)
- **Class 7**: Remission at 6 months, Remission at 12 months, Full symptomatic and functional recovery (Lowest risk)

This project addresses the **asymmetric costs of errors** in FEP prediction:

- Missing a potential relapse (false negative) can lead to devastating consequences (job loss, social isolation, self-harm, suicide)
- Unnecessary interventions (false positives) can cause side effects or other harms (such as unnecessary hospitalization, weight gain due to increased anti-psychotic dosage, etc.), but are generally less severe.

Thus, models and evaluation metrics have been specifically designed to prioritize the correct identification of high-risk cases.

## Key Features

1. **Cost-Sensitive Learning**:
   - Custom loss functions that penalize false negatives more heavily
   - Class weights based on clinical risk levels
   - Sample weighting to prioritize high-risk cases

2. **Threshold Optimization**:
   - Customized probability thresholds for each risk class
   - Lower thresholds for high-risk classes to increase sensitivity
   - Higher thresholds for low-risk classes to prevent overtreatment

3. **Clinical Utility Metrics**:
   - Custom evaluation metrics that incorporate asymmetric costs
   - Risk-adjusted AUC calculations
   - Time-weighted error metrics that penalize early misses more heavily

4. **Specialized Ensemble Models**:
   - High-Risk Focused Ensemble that combines general and specialized models
   - Time Decay Ensemble that incorporates time-based risk in predictions

5. **Early Warning Signs**:
   - Temporal feature extraction to detect deterioration trends
   - Warning sign identification based on clinical knowledge

6. **Interactive Dashboard**:
   - Risk-stratified prediction display
   - Clinical recommendations based on risk level
   - Model comparison with clinical interpretation

## Project Structure

```
fep_analysis/
├── config/                 # Configuration settings
│   ├── settings.py         # General settings
│   └── clinical_weights.py # Clinical risk weights and thresholds
├── data/                   # Data handling
│   ├── loader.py           # Data loading functions
│   └── preprocessor.py     # Data preprocessing
├── features/               # Feature engineering
│   ├── engineer.py         # Feature engineering
│   ├── selector.py         # Feature selection
│   └── temporal.py         # Temporal features for early warning
├── models/                 # ML models
│   ├── base.py             # Base model class with cost-sensitivity
│   ├── logistic.py         # Logistic regression model
│   ├── decision_tree.py    # Decision tree model
│   ├── gradient_boosting.py # Gradient boosting model
│   ├── ensemble.py         # Custom ensemble for high-risk cases
│   └── neural_network.py   # Neural network model
├── evaluation/             # Model evaluation
│   ├── metrics.py          # Custom evaluation metrics
│   ├── cross_validation.py # Cross-validation strategies
│   ├── threshold_optimization.py # Threshold optimization
│   └── clinical_utility.py # Clinical utility calculations
├── visualization/          # Visualization functions
│   ├── plots.py            # Basic visualization functions
│   ├── model_comparison.py # Model comparison visualizations
│   ├── feature_importance.py # Feature importance visualizations
│   └── risk_stratification.py # Risk stratification visualizations
└── webapp/                 # Interactive dashboard
    ├── app.py              # Main Streamlit application
    ├── pages/              # Dashboard pages
    └── components/         # UI components
```

## Installation and Usage

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fep-prediction.git
   cd fep-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard

```bash
cd webapp
streamlit run app.py
```

### Using the Models

```python
from models.ensemble import HighRiskFocusedEnsemble
from evaluation.metrics import clinical_utility_score
from evaluation.threshold_optimization import find_optimal_thresholds_multiclass

# Create and train a model
model = HighRiskFocusedEnsemble()
model.fit(X_train, y_train)

# Get predictions
y_pred = model.predict(X_test)

# Evaluate with clinical utility
utility, details = clinical_utility_score(y_test, y_pred)
print(f"Clinical utility: {utility:.3f}")

# Optimize thresholds
thresholds, results = find_optimal_thresholds_multiclass(y_test, model.predict_proba(X_test))
print("Optimized thresholds:", thresholds)
```

## Clinical Implications

The cost-sensitive approach in this project has several clinical implications:

1. **Risk Stratification** - Patients can be stratified by relapse risk to allocate resources appropriately

2. **Early Intervention Focus** - The models prioritize early detection of warning signs to enable timely intervention

3. **Transparent Decision Support** - The dashboard clearly communicates both predictions and their uncertainty

4. **Balanced Intervention** - The approach aims to balance the risks of under-treatment and over-treatment

## Literature Context

This project is informed by research on FEP outcomes and relapse prevention:

- Research by Leighton et al. (2019) demonstrating the ability for machine learning approaches to predict one-year outcomes in First Episode Psychosis 
- Research on Duration of Untreated Psychosis (DUP) that shows modest but meaningful association between shorter DUP and better outcomes

## Contributors

- Brence, Colin (updated project)
- Reybol, Antonio Jr.; Zhang, Xuewei; Zhang, Yilei (original CIS 9660 project)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

Leighton, S. P., Krishnadas, R., Chung, K., Blair, A., Brown, S., Clark, S., Sowerbutts, K., Schwannauer, M., Cavanagh, J., & Gumley, A. I. (2019). Predicting one-year outcome in first episode psychosis using machine learning. PLOS ONE, 14(3), e0212846. https://doi.org/10.1371/journal.pone.0212846
