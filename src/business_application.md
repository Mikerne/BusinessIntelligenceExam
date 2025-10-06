# Heart Disease Explorer - Business Application

## Project Overview

The **Heart Disease Explorer** is an interactive web application designed to make heart disease data analysis accessible to non-technical users. The application allows exploration, visualization, predictive modeling, and scenario testing using a **tuned Random Forest** model.

**Goals for Business Application:**
1. Create a web application with a simple visual interface.
2. Apply visual representations of data, analyses, and model results.
3. Provide explanation and interpretation of results for business users.

---

## Features of the App

### 1. Data Exploration (Tab: Data)
- Display dataset sample (up to 500 rows) with summary statistics.
- Show number of rows and columns.
- Provide statistical overview using `df.describe()`.

### 2. Exploratory Data Analysis (Tab: EDA)
- Correlation heatmap for numeric features.
- Interactive histograms for selected variables using Plotly.
- Enables users to observe distribution and relationships.

### 3. Model & Prediction (Tab: 🤖 Model & Prediction)
- Load or train a **tuned Random Forest** model with SMOTE and scaling.
- Users can adjust the **risk threshold** for 10-year CHD prediction.
  - Lower threshold → higher recall, more false positives.
  - Higher threshold → higher precision, fewer false positives.
- Display **test set metrics**:
  - ROC AUC, Accuracy, Precision, Recall, F1-score
- Feature importance visualization 
- Individual patient prediction form:
  - Users input patient features.
  - Receive personalized risk prediction (low, moderate, high risk).
- Optional SHAP explanations for model interpretability.

### 4. Clustering (Tab: 🌀 Clustering)
- Users can select 2–8 numeric features for KMeans clustering.
- Adjust the number of clusters interactively (2–6).
- Visualize clusters with color-coded **average CHD risk (%)**.
- Displays cluster-specific CHD risk alongside scatter plots.

### 5. Binned Analysis (Tab: 📈 Binned Analysis)
- Allows binning of numeric columns (3–10 bins).
- Computes mean of another selected column within each bin.
- Visualized as interactive bar charts for easy interpretation.

---

## Implementation Details

- **Framework**: Streamlit for interactive web interface.
- **Libraries**: pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, imbalanced-learn, joblib.
- **Model**: Tuned Random Forest
  - `n_estimators=300, max_depth=5, min_samples_split=5, min_samples_leaf=2, class_weight='balanced'`
- **Caching**: Streamlit caching for data and model to improve performance.
- **SHAP**: Optional for model interpretability (explains feature contributions).
