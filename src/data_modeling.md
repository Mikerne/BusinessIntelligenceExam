# Data Modelling - Heart Disease Prediction

## Project Description
This project applies AI methods and machine learning algorithms for predictive analytics on a heart disease dataset. The goal is to extend exploratory data analysis with predictive modelling, train models, and evaluate their performance for predicting "TenYearCHD".

This Sprint focuses on:
1. Selecting relevant machine learning methods (supervised learning).
2. Training and evaluating models on both training and test data.
3. Applying inference metrics to assess model quality.
4. Improving models through hyperparameter tuning.

## Dataset
- Input file: `../data/processed/heart_disease_clean_v2.csv`
- Contains heart-related features like age, gender, BMI, blood pressure, cholesterol, etc.
- Target variable: `TenYearCHD` (1 = develops CHD within 10 years, 0 = does not)

## Methods and Algorithms

### Classification
- **Logistic Regression**
  - Standardized numerical features
  - Handles imbalanced target with `SMOTE`
  - Metrics: Accuracy, Precision, Recall, F1, ROC AUC
- **Random Forest**
  - Tree-based ensemble classifier
  - Class weighting and SMOTE to handle imbalance
  - Feature importance visualization for top 10 features

### Hyperparameter Tuning
- **Tuned Random Forest**
  - Grid search parameters:
    ```python
    {
      'clf__n_estimators': [100, 200, 300],
      'clf__max_depth': [5, 10, None],
      'clf__min_samples_split': [2, 5],
      'clf__min_samples_leaf': [1, 2]
    }
    ```
  - StratifiedKFold (3 folds) cross-validation
  - Goal: Maximize ROC AUC on training data
  - Example results:
    - Best parameters: `{'clf__max_depth': 5, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__n_estimators': 100}`
    - Test set ROC AUC: ~0.66

## Evaluation Metrics
- **ROC AUC**: Measures classifier performance independent of threshold
- **Accuracy**: Fraction of correctly classified instances
- **Precision**: Fraction of predicted positives that are correct
- **Recall (Sensitivity)**: Fraction of actual positives correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual evaluation of predictions

## Conclusion

After training and evaluating the models on the Heart Disease dataset:

- **Logistic Regression**
  - CV ROC AUC: ~0.68
  - Test ROC AUC: ~0.63
  - High recall (~0.92), low precision (~0.17)  
  - Good at identifying most positive CHD cases, but many false positives

- **Random Forest**
  - CV ROC AUC: ~0.68
  - Test ROC AUC: ~0.62
  - Balanced recall (~0.78) and slightly higher precision (~0.19)  
  - Captures CHD cases fairly well, slightly better overall balance than Logistic Regression

- **Tuned Random Forest**
  - Best hyperparameters: `{'clf__max_depth': 5, 'clf__min_samples_leaf': 1, 'clf__min_samples_split': 2, 'clf__n_estimators': 100}`
  - Test ROC AUC: ~0.66
  - Recall: ~0.96, Precision: ~0.16
  - Maintains high sensitivity while slightly improving ROC AUC compared to untuned models

**Best Model:**  
The **Tuned Random Forest** is the best performing model for this dataset. It achieves the highest ROC AUC on the test set and maintains high recall, which is critical in healthcare settings where correctly identifying patients at risk of CHD is more important than avoiding false positives.  

Overall, hyperparameter tuning improved the model’s predictive ability and allowed for better feature importance interpretation, making it the most suitable choice for deployment and further analysis.
