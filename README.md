# ExamProject BusinessIntelligence  
**2025 Exam Project – Business Intelligence**  

## Predictive Analytics for Preventive Healthcare

### Annotation
This project addresses the challenge that many patients only receive treatment when their health issues have already progressed. Early intervention can significantly improve treatment outcomes and reduce healthcare costs.

By combining **Business Intelligence (BI)** and **Machine Learning (ML)**, we aim to analyze health and lifestyle data to develop predictive models that estimate the risk of diseases such as heart disease. The insights will help healthcare providers offer **proactive and personalized interventions**, ultimately supporting citizens to lead healthier lives.

---

### Problem Statement
Healthcare providers often lack systematic insights into which patients are at high risk for lifestyle-related diseases. By identifying the key predictive factors in patient demographics, lifestyle, and health records, providers can prioritize early interventions and preventive care.

---

### Context and Purpose
The healthcare system collects vast amounts of patient and lifestyle data, but it is rarely leveraged to predict risk and prevent disease. This project aims to build a **predictive BI solution** that enables:  

- Identification of patients at high risk.  
- Data-driven support for preventive measures.  
- Easy-to-understand dashboards and visualizations for non-technical healthcare staff.  

The goal is to empower healthcare professionals to make informed, proactive decisions, improving patient outcomes and optimizing healthcare resources.

---

### Research Questions
1. How can BI and ML be applied to analyze patient health and lifestyle data for preventive purposes?  
2. Which factors are most predictive of lifestyle-related disease risk?  
3. How can predictive models be integrated into a usable BI solution for healthcare providers?

---

### Hypotheses
- **H1:** A combination of demographic, lifestyle, and health data can predict increased risk of lifestyle-related diseases.  
- **H2:** Visualizations and dashboards help healthcare staff better understand risk patterns.  
- **H3:** Implementing predictive models supports early interventions and reduces future treatment costs.  

---

### Proposed Solution
We will build a **descriptive analytics and machine learning solution**:  

1. Perform an **exploratory data analysis (EDA)** to describe patterns in patient demographics, lifestyle, and health metrics.  
2. Develop **predictive models** to estimate disease risk based on patient data.  
3. Integrate the models into a **user-friendly BI dashboard using Streamlit**, enabling healthcare staff to explore risk predictions, patient profiles, and overall insights.  

**Key benefits:**  
- Identify high-risk patients early.  
- Enable targeted preventive care.  
- Provide actionable insights to healthcare providers.  

---

### Data and Scope
The project uses anonymized patient and lifestyle data with variables such as:  

- Age, BMI, blood pressure, cholesterol, glucose levels.  
- Smoking habits, medication usage, and diabetes status.  
- Disease outcome: **TenYearCHD** (10-year risk of coronary heart disease).  

**Data Processing:**  
Raw data is cleaned, processed, and transformed into a structured dataset suitable for modeling. Continuous variables are scaled, categorical variables are encoded, and missing values are handled.

---

### Project Planning
**Sprint Timeline:**  
- Sprint 1: Problem Formulation – 23/09/2025  
- Sprint 2: Data Preparation – 25/09/2025  
- Sprint 3: Data Modeling – 02/10/2025  
- Sprint 4: Business Application – 06/10/2025  

---

### Tools and Technologies
- **Streamlit** for dashboard and interactive BI application.  
- **Python** (Jupyter, pandas, scikit-learn) for data analysis and predictive modeling.  
- **Visualization Libraries:** Matplotlib, Seaborn, Plotly.  
- **Version Control:** GitHub.  

---

### Repository & Development Setup
**Git repository:** [link here]  

**Structure:**  
- `data/` – raw and processed datasets  
- `notebooks/` – experiments, report and analysis  
- `app_*.py` – Streamlit applications
- `src/` – Helpers and data import


**Software Requirements:**  
- Python 3.9+  
- Streamlit  
- scikit-learn, pandas, numpy, matplotlib, seaborn, plotly  

### How to Run the App
1. Clone the repository:  
https://github.com/Mikerne/BusinessIntelligenceExam)
streamlit run app.py
