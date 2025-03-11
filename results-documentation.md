# First Episode Psychosis (FEP) Outcome Analysis

## Executive Summary

This document presents results from the predictive modeling of remission outcomes in patients with First Episode Psychosis (FEP). The analysis aimed to identify reliable predictors of remission at 6 months, 12 months, and sustained remission, based on baseline demographic, clinical, and PANSS assessment data.

Three machine learning approaches were evaluated:
- Logistic Regression
- Decision Tree
- Gradient Boosting

The Gradient Boosting model achieved the highest predictive performance, with a weighted average AUC of 0.79 and overall accuracy of 72%. Key predictors identified include residual symptoms at early follow-up, employment status, delusions severity, and insight levels.

## Key Findings

### Outcome Patterns

Patients exhibited several distinct patterns of remission:

| Pattern | Description | Prevalence |
|---------|-------------|------------|
| Class 0 | No remission at 6 or 12 months | 19% |
| Class 1 | Sustained remission (at both 6 and 12 months) | 38% |
| Class 2 | Late remission (at 12 months only) | 22% |
| Class 3 | Early remission only (at 6 months, not sustained) | 11% |
| Class 6 | Other remission pattern | 10% |

### Predictive Factors

The analysis identified several key predictors of remission outcomes:

1. **Clinical Status Indicators**
   - Residual symptoms at early follow-up were the strongest predictors
   - Employment status showed significant predictive value

2. **PANSS Symptoms**
   - Positive symptoms, particularly delusions (P1) and suspiciousness (P6)
   - Lack of judgment and insight (G12)
   - Depression (G6)
   - Negative symptoms, particularly blunted affect (N1) and social withdrawal (N4)

3. **Demographic Factors**
   - Age and education level showed moderate importance
   - Accommodation type had some predictive value

## Model Performance

| Model | Weighted Avg AUC | Accuracy | Notes |
|-------|------------------|----------|-------|
| Gradient Boosting | 0.79 | 72% | Best overall performance |
| Logistic Regression | 0.76 | 68% | Good interpretability |
| Decision Tree | 0.71 | 64% | Simplest model |

Performance varied by outcome class:
- All models performed best for predicting Class 1 (sustained remission)
- Prediction of Class 3 (early non-sustained remission) was most challenging

## Clinical Implications

1. **Early Intervention Focus**
   - Monitoring residual symptoms at 6 months provides valuable prognostic information
   - Employment support may be particularly important for improving outcomes

2. **Symptom-Specific Treatment**
   - Addressing positive symptoms, particularly delusions, appears critical
   - Improving insight and addressing depression could improve remission chances

3. **Risk Stratification**
   - Patients can be stratified by remission risk to allocate resources appropriately
   - The model provides probability estimates for different outcome patterns

## Limitations and Future Directions

1. **Sample Size**
   - The analysis was based on a modest sample size
   - Some outcome classes had relatively few examples

2. **External Validation**
   - The models require validation in independent samples
   - Performance in different clinical settings needs confirmation

3. **Longitudinal Factors**
   - The analysis focused on baseline and early follow-up predictors
   - Integration of longitudinal data could further improve prediction

4. **Treatment Effects**
   - The current analysis does not account for treatment differences
   - Future work should examine the interaction between predictors and treatment approaches

## Implementation

The predictive models have been implemented in a user-friendly dashboard that allows clinicians to:
1. Enter patient data and obtain outcome probability estimates
2. Understand the reliability of predictions
3. Identify which factors are most influential for specific patients

This tool is intended to support clinical decision-making, not replace clinical judgment. The probabilistic predictions should be interpreted in the context of the individual patient and the clinical setting.
