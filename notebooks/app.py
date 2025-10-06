import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import joblib, os

# Optional: SHAP for feature explanations
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

st.set_page_config(page_title="❤️ Heart Disease Explorer", layout="wide")

# -------------------------
# Helper functions
# -------------------------
@st.cache_data
def load_data(path="../data/processed/heart_disease_clean_v2.csv"):
    return pd.read_csv(path)

@st.cache_resource
def train_or_load_tuned_model(df, target='TenYearCHD', cache_path='rf_heart_tuned.joblib', use_cache=True):
    """Load tuned Random Forest model with SMOTE, or train if not cached"""
    features_to_use = [
        'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 
        'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ]

    X = df[features_to_use]
    y = df[target]

    if use_cache and os.path.exists(cache_path):
        model = joblib.load(cache_path)
        scaler = joblib.load(cache_path + '.scaler')
        return model, scaler, True

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for balancing rare events
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scaling numeric features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_res)
    X_test_s = scaler.transform(X_test)

    # Tuned Random Forest
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_s, y_train_res)

    # Save model and scaler for reuse
    joblib.dump(model, cache_path)
    joblib.dump(scaler, cache_path + '.scaler')

    return model, scaler, False

def intervals_to_str(df):
    """Convert pandas Interval columns to strings for plotting"""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.IntervalDtype):
            df[col] = df[col].astype(str)
        else:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, pd.Interval) else x)
    return df

# -------------------------
# Main app
# -------------------------
st.title("❤️ Heart Disease Explorer")
st.markdown("""
This app allows healthcare professionals to:
- Explore patient dataset
- Analyze patterns in heart disease (EDA)
- Use a **tuned Random Forest model** to predict 10-year CHD risk
- Visualize patient clustering
- Explore binned relationships between variables
""")

df = load_data()

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Data", 
    "🔎 EDA", 
    "🤖 Model & Prediction", 
    "🌀 Clustering", 
    "📈 Binned Analysis"
])

# -------------------------
# Tab 1: Dataset
# -------------------------
with tab1:
    st.subheader("Dataset preview & summary")
    st.dataframe(df.sample(min(500, len(df))).reset_index(drop=True))
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    st.markdown("""
**Statistical summary:**  
Provides mean, standard deviation, min, max, and quartiles for each numeric variable.
""")

# -------------------------
# Tab 2: EDA
# -------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("#### 🔥 Correlation matrix")
    st.markdown("Shows how strongly pairs of numeric features are related. Values close to 1/-1 indicate strong positive/negative correlation.")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("#### 📦 Histograms")
    st.markdown("Shows the distribution of a selected variable. Box plot at the top helps identify outliers.")
    var = st.selectbox('Select variable', cols)
    fig2 = px.histogram(intervals_to_str(df), x=var, nbins=30, marginal="box")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Tab 3: Model & Prediction
# -------------------------
with tab3:
    st.subheader("Tuned Random Forest: Predict 10-year CHD risk")
    use_cache = st.checkbox("Use cached model if available", value=True)
    model, scaler, loaded = train_or_load_tuned_model(df, use_cache=use_cache)

    # Risk threshold slider
    st.markdown("#### ⚖️ Risk Threshold")
    st.markdown("""
Adjust the threshold probability for classifying a patient as 'at risk':  
- Lower threshold: captures more high-risk patients (higher recall) but may increase false positives  
- Higher threshold: reduces false positives but may miss some high-risk patients
""")
    risk_threshold = st.slider(
        "Select risk threshold",
        min_value=0.0, max_value=1.0, value=0.2, step=0.01
    )

    # Features for model
    features_to_use = [
        'male', 'age', 'currentSmoker', 'cigsPerDay', 'BPMeds', 
        'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 
        'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ]
    X = df[features_to_use]
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_s = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred_thresh = (y_proba >= risk_threshold).astype(int)

    # Metrics with explanations
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    roc = roc_auc_score(y_test, y_proba)

    st.markdown("#### 📊 Model Metrics (Test Set)")
    st.markdown(f"""
- **ROC AUC:** {roc:.3f} – Ability of model to distinguish patients with vs. without CHD  
- **Accuracy:** {acc:.3f} – Fraction of correct predictions overall  
- **Precision:** {prec:.3f} – Of all predicted at risk, how many actually develop CHD  
- **Recall:** {rec:.3f} – Of all patients who develop CHD, how many were correctly predicted  
- **F1-score:** {f1:.3f} – Balance between Precision and Recall
""")

    # Feature importance with explanation
    st.markdown("#### 📌 Feature importance")
    st.markdown("Shows which factors contributed most to predicting 10-year CHD risk.")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances.head(15))

    # Individual patient prediction
    st.markdown("#### 👤 Individual patient prediction")
    st.markdown("Input patient data to get a personalized CHD risk prediction.")
    input_row = {}
    with st.form("predict_form"):
        for c in X.columns:
            input_row[c] = st.number_input(c, value=float(df[c].median()))
        submitted = st.form_submit_button("Predict")

    if submitted:
        X_new = pd.DataFrame([input_row])
        X_new_s = scaler.transform(X_new)
        prob = model.predict_proba(X_new_s)[0, 1]
        y_new_pred = (prob >= risk_threshold).astype(int)

        if y_new_pred == 0:
            st.success(f"🟢 Low risk ({prob:.1%})")
        elif y_new_pred == 1 and prob < 0.5:
            st.warning(f"🟡 Moderate risk ({prob:.1%})")
        else:
            st.error(f"🔴 High risk ({prob:.1%})")

# -------------------------
# Tab 4: Clustering
# -------------------------
with tab4:
    st.subheader("🌀 Clustering with custom features")
    st.markdown("Group patients based on selected features to explore patterns in CHD risk.")

    numeric = df.select_dtypes(include=[np.number]).drop(columns=['TenYearCHD'])
    selected_features = st.multiselect(
        "Select features for clustering (2-8)",
        options=numeric.columns.tolist(),
        default=['age', 'totChol'],
    )
    
    if len(selected_features) >= 2:
        cluster_k = st.slider("Number of clusters (KMeans)", 2, 6, 2)
        scaler_c = StandardScaler()
        X_scaled = scaler_c.fit_transform(df[selected_features])
        km = KMeans(n_clusters=cluster_k, random_state=42)
        labels = km.fit_predict(X_scaled)
        df_plot = df[selected_features].copy()
        df_plot['Cluster'] = labels
        df_plot['TenYearCHD'] = df['TenYearCHD']
        cluster_chd_mean = df_plot.groupby('Cluster')['TenYearCHD'].mean().to_dict()
        df_plot['ClusterRisk'] = df_plot['Cluster'].map(cluster_chd_mean) * 100

        fig = px.scatter(
            df_plot,
            x=selected_features[0],
            y=selected_features[1],
            color='ClusterRisk',
            color_continuous_scale='RdYlBu_r',
            hover_data=selected_features + ['ClusterRisk'],
            title=f"KMeans clustering ({cluster_k} clusters) - Average CHD risk (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### 📊 Cluster risk (%)")
        st.markdown("Shows the average 10-year CHD risk for patients in each cluster.")
        for c in range(cluster_k):
            st.write(f"Cluster {c}: {cluster_chd_mean[c]*100:.1f}% risk")
    else:
        st.warning("Select at least 2 features for clustering")

# -------------------------
# Tab 5: Binned analysis
# -------------------------
with tab5:
    st.subheader("Binned analysis")
    st.markdown("Compare how the average of a variable changes across intervals of another variable.")
    numeric = df.select_dtypes(include=[np.number])
    bin_col = st.selectbox("Column for binning", numeric.columns)
    agg_col = st.selectbox("Column to aggregate", numeric.columns)
    nbins = st.slider("Number of bins", 3, 10, 5)

    df_bins = intervals_to_str(df.copy())
    df_bins['bins'] = pd.cut(df_bins[bin_col], bins=nbins).astype(str)
    agg = df_bins.groupby('bins', observed=True)[agg_col].mean().reset_index(name=f"mean_{agg_col}")

    fig = px.bar(
        agg,
        x='bins',
        y=f"mean_{agg_col}",
        title=f"Average of {agg_col} per {bin_col} interval"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("Run the app locally with: `streamlit run streamlit_app.py`")
