import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import joblib, os

# Optional: SHAP
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
    """Load tuned Random Forest model with SMOTE from notebook, or train if not cached"""
    if use_cache and os.path.exists(cache_path):
        model = joblib.load(cache_path)
        scaler = joblib.load(cache_path + '.scaler')
        return model, scaler, True

    X = df.drop(columns=[target])
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE for balancing
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_res)
    X_test_s = scaler.transform(X_test)

    # Tuned hyperparameters from notebook
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

    # Save model and scaler
    joblib.dump(model, cache_path)
    joblib.dump(scaler, cache_path + '.scaler')

    return model, scaler, False

def intervals_to_str(df):
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
Denne app giver dig mulighed for at:
- Udforske datasættet
- Analysere mønstre i hjertesygdom (EDA)
- Træne & bruge en **tuned Random Forest-model** til at forudsige risiko
- Visualisere clustering og PCA
- Afprøve patient-scenarier med individuelle prædiktioner
""")

df = load_data()

# Tabs
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
    st.write("**Statistisk oversigt:**")
    st.write(df.describe())

# -------------------------
# Tab 2: EDA
# -------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("#### 🔥 Korrelationsmatrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("#### 📦 Histogrammer")
    var = st.selectbox('Vælg variabel', cols)
    fig2 = px.histogram(intervals_to_str(df), x=var, nbins=30, marginal="box")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Tab 3: Model & Prediction
# -------------------------
with tab3:
    st.subheader("Tuned Random Forest: Predict 10-year CHD")
    use_cache = st.checkbox("Brug gemt model, hvis tilgængelig", value=True)
    model, scaler, loaded = train_or_load_tuned_model(df, use_cache=use_cache)

    # Risk threshold slider
    st.markdown("#### ⚖️ Risk Threshold")
    risk_threshold = st.slider(
        "Juster risk threshold for 10-year CHD prediction",
        min_value=0.0, max_value=1.0, value=0.2, step=0.01
    )
    st.markdown("""
    **Forklaring:**  
    - Lav threshold (fx 0.2) fanger flere med høj risiko (høj recall), men kan give flere falsk positive (lav precision).  
    - Høj threshold (fx 0.5) giver færre falsk positive (højere precision), men kan overse nogle sande tilfælde (lavere recall).  
    Anbefalet standardværdi: **0.2**
    """)

    # Test metrics
    X = df.drop(columns=['TenYearCHD'])
    y = df['TenYearCHD']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_s = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred_thresh = (y_proba >= risk_threshold).astype(int)

    # Calculate all metrics
    acc = accuracy_score(y_test, y_pred_thresh)
    prec = precision_score(y_test, y_pred_thresh)
    rec = recall_score(y_test, y_pred_thresh)
    f1 = f1_score(y_test, y_pred_thresh)
    roc = roc_auc_score(y_test, y_proba)

    st.markdown("#### 📊 Model Metrics (Test Set)")
    st.write(f"**ROC AUC:** {roc:.3f} – Evnen til at skelne mellem patienter med og uden sygdom")
    st.write(f"**Accuracy:** {acc:.3f} – Andel korrekt klassificerede tilfælde")
    st.write(f"**Precision:** {prec:.3f} – Andel korrekt positive forudsigelser ud af alle positive forudsigelser")
    st.write(f"**Recall:** {rec:.3f} – Andel sande positive tilfælde, der blev fanget af modellen")
    st.write(f"**F1-score:** {f1:.3f} – Harmonisk gennemsnit af Precision og Recall")

    # Feature importance
    st.markdown("#### 📌 Feature importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    st.bar_chart(importances.head(15))

    # Individual patient prediction
    st.markdown("#### 👤 Individuel patient-prædiktion")
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
            st.success(f"🟢 Lav risiko ({prob:.1%})")
        elif y_new_pred == 1 and prob < 0.5:
            st.warning(f"🟡 Moderat risiko ({prob:.1%})")
        else:
            st.error(f"🔴 Høj risiko ({prob:.1%})")

        if SHAP_AVAILABLE:
            st.markdown("#### SHAP forklaring")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_new_s)
            try:
                shap_html = shap.force_plot(
                    explainer.expected_value[1], shap_values[1], X_new, matplotlib=False
                )
                st.components.v1.html(shap_html.html(), height=400)
            except Exception:
                st.info("Kunne ikke vise SHAP plot i dette miljø")

# -------------------------
# Tab 4: Clustering
# -------------------------
with tab4:
    st.subheader("🌀 Clustering med brugerdefinerede features")
    
    # Vælg features til clustering
    numeric = df.select_dtypes(include=[np.number]).drop(columns=['TenYearCHD'])
    selected_features = st.multiselect(
        "Vælg features til clustering (2-8)",
        options=numeric.columns.tolist(),
        default=['age', 'totChol'],
    )
    
    if len(selected_features) < 2:
        st.warning("Vælg mindst 2 features for at lave clustering")
    else:
        # Antal clusters
        cluster_k = st.slider("Antal clusters (KMeans)", 2, 6, 2)

        # Skaler data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[selected_features])

        # KMeans clustering
        km = KMeans(n_clusters=cluster_k, random_state=42)
        labels = km.fit_predict(X_scaled)

        df_plot = df[selected_features].copy()
        df_plot['Cluster'] = labels
        df_plot['TenYearCHD'] = df['TenYearCHD']

        # Beregn gennemsnitlig TenYearCHD pr. cluster i procent
        cluster_chd_mean = df_plot.groupby('Cluster')['TenYearCHD'].mean().to_dict()
        df_plot['ClusterRisk'] = df_plot['Cluster'].map(cluster_chd_mean) * 100  # procent

        # Plot: brug de første to features som x og y
        fig = px.scatter(
            df_plot,
            x=selected_features[0],
            y=selected_features[1],
            color='ClusterRisk',
            color_continuous_scale='RdYlBu_r',
            hover_data=selected_features + ['ClusterRisk'],
            title=f"KMeans clustering ({cluster_k} clusters) - farve = gennemsnitlig CHD risiko (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Vis cluster-risiko ved siden af
        st.markdown("#### 📊 Cluster risiko (%)")
        for c in range(cluster_k):
            st.write(f"Cluster {c}: {cluster_chd_mean[c]*100:.1f}% risiko")





# -------------------------
# Tab 5: Binned analysis
# -------------------------
with tab5:
    st.subheader("Binned analysis")
    numeric = df.select_dtypes(include=[np.number])
    bin_col = st.selectbox("Kolonne til binning", numeric.columns)
    agg_col = st.selectbox("Kolonne til gennemsnit", numeric.columns)
    nbins = st.slider("Antal bins", 3, 10, 5)

    df_bins = intervals_to_str(df.copy())
    df_bins['bins'] = pd.cut(df_bins[bin_col], bins=nbins).astype(str)
    agg = df_bins.groupby('bins', observed=True)[agg_col].mean().reset_index(name=f"mean_{agg_col}")

    fig = px.bar(
        agg,
        x='bins',
        y=f"mean_{agg_col}",
        title=f"Gennemsnit af {agg_col} pr. {bin_col}-interval"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.info("Kør appen lokalt med: `streamlit run streamlit_app_sprint4_userfriendly_fixed.py`")
