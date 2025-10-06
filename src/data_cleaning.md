# Heart Disease Dataset – Data Cleaning Documentation

## Overview
This README documents the data cleaning and preprocessing steps applied to the `heart_disease.csv` dataset. The purpose of this cleaning is to prepare a high-quality, structured dataset for **Business Intelligence (BI) analysis** and **predictive modeling** in preventive healthcare.

The dataset contains patient demographics, lifestyle, and health metrics, including the target variable `TenYearCHD` (10-year risk of coronary heart disease).

---

## Dataset Structure
- **Raw Data:** `../data/raw/heart_disease.csv`
- **Processed Data:** `../data/processed/heart_disease_clean_v2.csv`
- **Number of Rows:** 4238 (after cleaning)
- **Selected Columns:**
  - `male`, `age`, `education`, `currentSmoker`, `cigsPerDay`,  
  - `BPMeds`, `prevalentStroke`, `prevalentHyp`, `diabetes`,  
  - `totChol`, `sysBP`, `diaBP`, `BMI`, `heartRate`, `glucose`, `TenYearCHD`

---

## Data Cleaning Steps

### 1. Column Selection
Only columns relevant to predictive modeling and BI analysis were retained to ensure **data relevance**.

### 2. Handling Missing Values
- **Numerical columns:** Missing values were replaced with the **median** of each column.
- **Categorical columns (`education`, `BPMeds`):** Missing values were replaced with the **mode**.

### 3. Outlier Capping
- Values outside the **1st and 99th percentile** for numerical columns were capped.
- Ensures **robustness** of statistical analyses and predictive models.

### 4. Verification
- Checked for remaining missing values to guarantee **data completeness**.
- `assert df.isna().sum().sum() == 0` ensures no NA values remain.

### 5. Save Processed Dataset
- Cleaned dataset saved as: `../data/processed/heart_disease_clean_v2.csv`
- Ready for **EDA, visualization, and machine learning models**.

---

## Optional Insights
- Class distribution for target variable `TenYearCHD`:
```text
0: 84%
1: 16%
