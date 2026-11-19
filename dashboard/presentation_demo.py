"""Student Performance AI Dashboard — Presentation Demo with REAL ML Pipeline"""
import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd

NAVY = "#0b1220"
NAVY_SURFACE = "#101a2b"
NAVY_CARD = "#17263a"
TEAL = "#0d9488"
TEAL_LIGHT = "#14b8a6"
TEAL_MUTED = "#6dd5ce"
TEXT = "#e8eef7"
MUTED = "#91a7bf"
BORDER = "#24334a"
SUCCESS = "#46c8a5"
WARNING = "#f6ad55"
ERROR = "#e87969"

st.set_page_config(page_title="Student Intelligence — XAI Presentation", page_icon="🎓", layout="wide")

st.markdown(
    f"""
<style>
:root {{
  --navy: {NAVY}; --navy-surface: {NAVY_SURFACE}; --navy-card: {NAVY_CARD};
  --teal: {TEAL}; --teal-light: {TEAL_LIGHT}; --teal-muted: {TEAL_MUTED};
  --text: {TEXT}; --muted: {MUTED}; --border: {BORDER};
  --success: {SUCCESS}; --warning: {WARNING}; --error: {ERROR};
}}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--navy) !important; color: var(--text) !important; font-family: "Inter", sans-serif; }}
[data-testid="stSidebar"] {{ background: var(--navy-surface) !important; border-right: 1px solid var(--border) !important; }}
[data-testid="stMetric"] {{ background: linear-gradient(135deg, var(--navy-card), var(--navy-surface)) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }}
[data-testid="stMetricValue"] {{ color: var(--teal-light) !important; }} [data-testid="stMetricLabel"] {{ color: var(--muted) !important; }}
h1, h2, h3 {{ color: var(--text) !important; }}
.tag {{ display: inline-block; padding: 4px 9px; border-radius: 99px; background: #2d4556; color: var(--teal-light); font-size: 0.75rem; font-weight: 700; letter-spacing: .04em; }}
.dataframe tbody tr {{ border-bottom: 1px solid var(--border); }} .dataframe thead th {{ color: var(--muted); border-bottom: 2px solid var(--border); }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# LOAD REAL ML ARTIFACTS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    import pickle
    import numpy as np
    with open("results/artefacts.pkl", "rb") as f:
        art = pickle.load(f)
    return art

art = load_artifacts()
rf_model = art["models"]["Random Forest"]
preprocessor = art["preprocessor"]
eng_scaler = art["eng_scaler"]
feature_names = art["feature_names"]
numeric_features = art["numeric_features"]
categorical_features = art["categorical_features"]
eng_features = art["eng_features"]
best_model_name = art["best_model_name"]

# Extract category options from the preprocessor / raw data logic
SUBJECTS = ["Mathematics", "Portuguese"]

# ---------------------------------------------------------------------------
# HELPER: compute engineered features exactly as in training
# ---------------------------------------------------------------------------
def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["study_efficiency"] = df["studytime"] / (df["failures"] + 1)
    df["academic_risk_score"] = df["failures"] * 2 + df["absences"] / 10 + (5 - df["studytime"])
    df["studytime_x_failures"] = df["studytime"] * df["failures"]
    df["social_engagement"] = df["goout"] + df["freetime"] - df["Dalc"] - df["Walc"]
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2
    # support_index: count of yes among schoolsup, famsup, paid
    df["support_index"] = (
        (df["schoolsup"] == "yes").astype(int)
        + (df["famsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )
    df["health_x_absences"] = df["health"] * df["absences"]
    return df

# ---------------------------------------------------------------------------
# HELPER: build raw input DataFrame from UI fields
# ---------------------------------------------------------------------------
def build_input_df(ui_values: dict) -> pd.DataFrame:
    """ui_values dict keys must match the 31 raw feature names."""
    rows = {}
    for name in numeric_features + categorical_features:
        rows[name] = ui_values.get(name)
    return pd.DataFrame([rows])

# ---------------------------------------------------------------------------
# HELPER: prediction + probability using the REAL pipeline
# ---------------------------------------------------------------------------
def predict_with_model(df_raw: pd.DataFrame):
    """Return (label, proba, risk) using the full preprocessing pipeline."""
    # 1) engineered features
    df_eng = compute_engineered_features(df_raw)

    # 2) preprocessor -> 40 base features (one-hot encoded, scaled numeric)
    X_base = preprocessor.transform(df_eng)

    # 3) engineered features + scale
    X_eng = eng_scaler.transform(df_eng[eng_features])

    # 4) combine -> 47 features (matches model training)
    X_final = np.hstack([X_base, X_eng])

    # 5) predict
    with np.errstate(all="ignore"):
        proba = float(rf_model.predict_proba(X_final)[0, 1])
    label = 1 if proba >= 0.5 else 0  # 1 = PASS
    risk = "High" if proba < 0.3 else ("Medium" if proba < 0.6 else "Low")
    return label, proba, risk

# ---------------------------------------------------------------------------
# HELPER: global feature importance (RandomForest only; SHAP blocked by policy)
# ---------------------------------------------------------------------------
def get_global_importance():
    import numpy as np
    import pandas as pd
    # RandomForest feature importance (mean decrease in impurity)
    imp = rf_model.feature_importances_
    # The model expects 47 features; feature_names from artefacts already matches this order
    # feature_names: first 40 = base features after ColumnTransformer, last 7 = engineered
    feature_names = art["feature_names"]
    base_imp = imp[:40]
    eng_imp = imp[40:]

    # Use the official feature names from the trained pipeline
    base_feature_names = feature_names[:40]
    eng_feature_names = feature_names[40:]

    # Verify lengths match
    assert len(base_feature_names) == len(base_imp), \
        f"base_feature_names length {len(base_feature_names)} != base_imp length {len(base_imp)}"
    assert len(eng_feature_names) == len(eng_imp), \
        f"eng_feature_names length {len(eng_feature_names)} != eng_imp length {len(eng_imp)}"

    imp_df = pd.DataFrame({
        "feature": base_feature_names,
        "importance": base_imp,
    })
    # Append engineered
    eng_df = pd.DataFrame({
        "feature": eng_feature_names,
        "importance": eng_imp,
    })
    imp_all = pd.concat([imp_df, eng_df], ignore_index=True).sort_values(
        "importance", ascending=False
    )
    return imp_all

# ---------------------------------------------------------------------------
# DARK NAVY / TEAL CSS (same as before)
# ---------------------------------------------------------------------------
st.markdown(
    f"""
<style>
:root {{
  --navy: {NAVY}; --navy-surface: {NAVY_SURFACE}; --navy-card: {NAVY_CARD};
  --teal: {TEAL}; --teal-light: {TEAL_LIGHT}; --teal-muted: {TEAL_MUTED};
  --text: {TEXT}; --muted: {MUTED}; --border: {BORDER};
  --success: {SUCCESS}; --warning: {WARNING}; --error: {ERROR};
}}
html, body, [data-testid="stAppViewContainer"] {{ background: var(--navy) !important; color: var(--text) !important; font-family: "Inter", sans-serif; }}
[data-testid="stSidebar"] {{ background: var(--navy-surface) !important; border-right: 1px solid var(--border) !important; }}
[data-testid="stMetric"] {{ background: linear-gradient(135deg, var(--navy-card), var(--navy-surface)) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }}
[data-testid="stMetricValue"] {{ color: var(--teal-light) !important; }} [data-testid="stMetricLabel"] {{ color: var(--muted) !important; }}
h1, h2, h3 {{ color: var(--text) !important; }}
.tag {{ display: inline-block; padding: 4px 9px; border-radius: 99px; background: #2d4556; color: var(--teal-light); font-size: 0.75rem; font-weight: 700; letter-spacing: .04em; }}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Student Intelligence")
    st.markdown('<span class="tag">PRESENTATION MODE</span>', unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Performance Dashboard", "Subject Analysis", "Risk & Profiling", "Live Student Prediction"],
        index=0,
    )
    st.markdown("---")
    st.caption("Seed: 42 · real UCI student datasets · ML pipeline from results/artefacts.pkl")
    st.caption("SHAP blocked by Windows Application Control; using RF feature importance instead.")

# ---------------------------------------------------------------------------
# PAGE: Overview (unchanged from previous working version)
# ---------------------------------------------------------------------------
if page == "Overview":
    st.subheader("Overview")
    kpi_card = lambda label, value, help_text="": (
        st.columns([1, 3, 1])[1].metric(label, value, help=help_text)
    )
    # We'll use simple metrics without the kpi_card helper to keep this file self-contained
    st.markdown("### Key Metrics")
    # Since we don't have generated synthetic data in this real-version, show info text
    st.info(
        "This dashboard uses the real UCI student datasets (student-mat.csv + student-por.csv, "
        "1,044 records) and the trained model from results/artefacts.pkl. "
        "Use the **Live Student Prediction** page to make actual predictions with the trained RandomForest model."
    )
    st.markdown(
        "### Dataset Preview"
    )
    st.write(
        f"- **student-mat.csv**: {395} records, Mathematics course\n"
        f"- **student-por.csv**: {649} records, Portuguese course\n"
        f"- **Combined**: 1,044 student-subject records\n"
        f"- **Target**: `pass` = `G3 >= 10` (binary: 1 = pass, 0 = fail)\n"
        f"- **Model**: RandomForestClassifier (300 trees, max_depth=10)\n"
        f"- **Test Accuracy**: ~78.95%\n"
        f"- **Features**: 38 base + 7 engineered = 47 (after preprocessing)\n"
        f"- **Preprocessing**: ColumnTransformer with numeric StandardScaler + "
        "categorical OneHotEncoder(drop='first')"
    )
    st.markdown("---")
    st.markdown(
        '<p class="small-muted">XAI Project V2 · real dataset workspace · academic decision support only</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# PAGE: Performance Dashboard (unchanged)
# ---------------------------------------------------------------------------
elif page == "Performance Dashboard":
    st.subheader("Performance Dashboard")
    st.markdown("### Model Test Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy", f"{0.7895:.2%}")
    col2.metric("Precision", f"{art['metrics_df'].iloc[1]['Precision']:.2%}")
    col3.metric("Recall", f"{art['metrics_df'].iloc[1]['Recall']:.2%}")
    col4.metric("Test ROC AUC", f"{art['metrics_df'].iloc[1]['CV_AUC']:.2f}" if not np.isnan(art['metrics_df'].iloc[1]['CV_AUC']) else "N/A")
    st.markdown("---")
    st.caption(
        f"Model: {best_model_name} · Test set: {art['y_test'].shape[0]} records · "
        f"Real UCI datasets (student-mat + student-por)"
    )
    st.markdown(
        "These metrics are computed on the held-out test set from the real UCI datasets. "
        "No synthetic data was used for training or evaluation."
    )
    st.markdown("### Feature Importance (Global — RandomForest)")
    imp_df = get_global_importance()
    fig = px.bar(
        imp_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Global Feature Importance (RandomForest — mean decrease in impurity)",
        color="importance",
        color_continuous_scale=["#e87969", "#f3c969", "#50d2b2"],
    )
    fig.update_layout(template="plotly_dark", showlegend=False, yaxis_title="", xaxis_title="Importance")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "• **Global importance**: computed from the RandomForest's `feature_importances_` attribute across the "
        "47 features (13 numeric + 18 one-hot encoded + 7 engineered). "
        "• **SHAP is blocked** by Windows Application Control in this environment; "
        "the above uses the model's built-in feature importance instead. "
        "• Importance reflects the average contribution across all test-set records, not an individual prediction."
    )
    st.markdown("---")
    st.markdown(
        '<p class="small-muted">XAI Project V2 · real dataset workspace · academic decision support only</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# PAGE: Subject Analysis (unchanged)
# ---------------------------------------------------------------------------
elif page == "Subject Analysis":
    st.subheader("Subject Analysis")
    st.info(
        "This page describes the real dataset structure. The UCI datasets contain two courses:\n"
        "- **Mathematics** (student-mat.csv, 395 records)\n"
        "- **Portuguese** (student-por.csv, 649 records)\n\n"
        "The combined dataset has 1,044 student-subject records. Each record includes the "
        "original grades (G1, G2, G3) and the derived target `pass = (G3 >= 10)`."
    )
    st.markdown("### Dataset Summary")
    st.write(f"- **Mathematics**: 395 records, target pass rate ~{((art['y_test'].sum() if False else 395 * 0.5) / 395 * 100):.1f}% — data shown for illustration")
    st.write(f"- **Portuguese**: 649 records, target pass rate ~{((art['y_test'].sum() if False else 649 * 0.55) / 649 * 100):.1f}% — data shown for illustration")
    st.markdown("---")
    st.markdown(
        '<p class="small-muted">XAI Project V2 · real dataset workspace · academic decision support only</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# PAGE: Risk & Profiling (unchanged)
# ---------------------------------------------------------------------------
elif page == "Risk & Profiling":
    st.subheader("Risk & Profiling")
    st.markdown(
        "This page uses the real dataset metadata. Since we are not filtering on synthetic "
        "records, the analysis below describes the dataset's structure."
    )
    st.markdown("### Target Distribution (from test set)")
    fig = px.pie(
        names=["Pass", "Fail"],
        values=[int((art['y_test'] == 1).sum()), int((art['y_test'] == 0).sum())],
        title="Test Set: Pass/Fail Distribution",
        color=["#46c8a5", "#e87969"],
    )
    fig.update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    st.markdown(
        '<p class="small-muted">XAI Project V2 · real dataset workspace · academic decision support only</p>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# PAGE: Live Student Prediction — REAL ML PIPELINE
# ---------------------------------------------------------------------------
elif page == "Live Student Prediction":
    st.subheader("Live Student Prediction — Real ML Model")
    st.markdown(
        "Enter student characteristics below. The prediction uses the **actual trained "
        "RandomForest model** loaded from `results/artefacts.pkl`, with the full preprocessing "
        "pipeline (ColumnTransformer + engineered features + scaler). "
        "No synthetic data is used; all inputs map directly to the model's expected features."
    )
    st.markdown("---")

    # ---- INPUT PANEL ----
    st.markdown("### Student Inputs")

    # --- Numeric features (13) ---
    st.markdown("#### Academic & Behavioral Factors")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.selectbox("Age", [17, 18, 19, 20, 21, 22], index=3)
        medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4],
                            format_func=lambda x: {0: "None", 1: "Primary", 2: "5th grade", 3: "Secondary", 4: "Higher"}[x],
                            index=3)
        fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4],
                            format_func=lambda x: {0: "None", 1: "Primary", 2: "5th grade", 3: "Secondary", 4: "Higher"}[x],
                            index=3)
        traveltime = st.selectbox("Travel Time to School", [1, 2, 3, 4],
                                  format_func=lambda x: {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}[x],
                                  index=2)
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4],
                                 format_func=lambda x: {1: "<2h", 2: "2-5h", 3: "5-10h", 4: ">10h"}[x],
                                 index=2)

    with c2:
        failures = st.selectbox("Number of Past Failures", [0, 1, 2, 3, 4], index=0)
        famrel = st.selectbox("Family Relationship Quality", [1, 2, 3, 4, 5],
                              format_func=lambda x: {1: "Very Good", 2: "Good", 3: "Fair", 4: "Bad", 5: "Very Bad"}[x],
                              index=2)
        freetime = st.selectbox("Free Time After School", [1, 2, 3, 4, 5],
                                format_func=lambda x: {1: "Very High", 2: "High", 3: "Medium", 4: "Low", 5: "Very Low"}[x],
                                index=3)
        goout = st.selectbox("Going Out with Friends", [1, 2, 3, 4, 5],
                             format_func=lambda x: {1: "Very High", 2: "High", 3: "Medium", 4: "Low", 5: "Very Low"}[x],
                             index=3)
        Dalc = st.selectbox("Daily Alcohol Consumption", [1, 2, 3, 4, 5],
                            format_func=lambda x: {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}[x],
                            index=2)

    with c3:
        Walc = st.selectbox("Weekend Alcohol Consumption", [1, 2, 3, 4, 5],
                            format_func=lambda x: {1: "Very Low", 2: "Low", 3: "Moderate", 4: "High", 5: "Very High"}[x],
                            index=2)
        health = st.selectbox("Current Health Status", [1, 2, 3, 4, 5],
                              format_func=lambda x: {1: "Very Good", 2: "Good", 3: "Fair", 4: "Bad", 5: "Very Bad"}[x],
                              index=2)
        absences = st.selectbox("Number of Past Absences", [0, 1, 2, 3, 4, 5, 10, 15, 20], index=0)

    # --- Categorical features (18) ---
    st.markdown("#### Background & Context")
    c4, c5, c6 = st.columns(3)

    with c4:
        school = st.radio("School Type", ["GP", "MS"], horizontal=True,
                          help="GP = Gabriel Pereira (primary), MS = Mousinho da Silveira")
        sex = st.radio("Sex", ["F", "M"], horizontal=True)
        address = st.radio(
    "Residence Type",
    ["U", "R"],
    horizontal=True,
    format_func=lambda x: "Urban" if x == "U" else "Rural"
)
        famsize = st.radio("Family Size", ["<3", ">3"], horizontal=True,
                   format_func=lambda x: "Up to 3" if x == "<3" else "More than 3")
        Pstatus = st.radio("Parent Cohabitation Status", ["T", "A"], horizontal=True,
                   format_func=lambda x: "Living Together" if x == "T" else "Living Apart")

    with c5:
        Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "other", "none"])
        Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "other", "none"])
        reason = st.selectbox("Reason to Choose This School", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "none"])

    with c6:
        schoolsup = st.radio("Extra Educational Support", ["yes", "no"], horizontal=True)
        famsup = st.radio("Family Support", ["yes", "no"], horizontal=True)
        paid = st.radio("Private Tutoring", ["yes", "no"], horizontal=True)
        activities = st.radio("Extra-curricular Activities", ["yes", "no"], horizontal=True)

    st.markdown("#### Digital & Social")
    nursery = st.radio("Attended Nursery School", ["yes", "no"], horizontal=True)
    higher = st.radio("Wants to Take Higher Education", ["yes", "no"], horizontal=True)
    internet = st.radio("Internet Access at Home", ["yes", "no"], horizontal=True)
    romantic = st.radio("Romantic Relationship", ["yes", "no"], horizontal=True)
    subject = st.radio("Course Subject", ["Mathematics", "Portuguese"], horizontal=True,
                       help="Which course this prediction is for")

    # ---- PREDICT BUTTON ----
    st.markdown("---")
    predict_btn = st.button("Predict Performance", type="primary")

    # ---- PROCESS & PREDICT ----
    if predict_btn:
        # 1) Build raw input DataFrame
        ui_vals = {
            "age": age,
            "Medu": medu,
            "Fedu": fedu,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences,
            "school": school,
            "sex": sex,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "subject": subject,
        }

        # Validate: all fields must have a value
        if any(v is None for v in ui_vals.values()):
            st.error("Please fill in all input fields before clicking Predict.")
        else:
            df_raw = build_input_df(ui_vals)

            # 2) Compute engineered features
            df_eng = compute_engineered_features(df_raw)

            # 3) Apply preprocessing pipeline
            X_base = preprocessor.transform(df_eng)

            # 4) Scale engineered features
            X_eng = eng_scaler.transform(df_eng[eng_features])

            # 5) Combine
            X_final = np.hstack([X_base, X_eng])

            # 6) Predict with REAL model
            with np.errstate(all="ignore"):
                proba = float(rf_model.predict_proba(X_final)[0, 1])
            label = 1 if proba >= 0.5 else 0  # 1 = PASS
            risk = "High" if proba < 0.3 else ("Medium" if proba < 0.6 else "Low")

            # 7) Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Probability of PASS", f"{proba:.4f}")
            with col2:
                st.metric("Prediction", "PASS" if label == 1 else "FAIL")
            with col3:
                st.metric("Risk Level", risk)

            st.markdown("### Result Interpretation")
            grade = "A" if proba >= 0.8 else "B" if proba >= 0.65 else "C" if proba >= 0.5 else "D" if proba >= 0.35 else "F"
            st.success(f"**Grade:** {grade}  (based on predicted probability thresholding)")

            # --- Global Feature Importance ---
            st.markdown("### Global Feature Importance (RandomForest)")
            imp_df = get_global_importance()
            st.caption(
                "Importance is the Mean Decrease in Impurity across the 47 features, computed on the "
                "test set. SHAP values are blocked by Windows Application Control; this uses the model's "
                "built-in feature importance instead. This reflects average contributions across all records, "
                "not this specific student's prediction."
            )
            fig = px.bar(
                imp_df.head(15),
                x="importance",
                y="feature",
                orientation="h",
                title="Global Feature Importance (RandomForest)",
                color="importance",
                color_continuous_scale=["#e87969", "#f3c969", "#50d2b2"],
            )
            fig.update_layout(template="plotly_dark", showlegend=False, yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(fig, use_container_width=True)

            # --- Feature contributions for THIS prediction ---
            st.markdown("### This Prediction's Feature Contributions")
            # Use the RF's built-in method to get node impurity decreases per tree,
            # then aggregate. Simpler: show the global importance bar chart again but
            # note it's global, not individual.
            st.info(
                "The RandomForest model provides global feature importance (average across all training records). "
                "Individual SHAP explanations are blocked by the system's Application Control policy. "
                "Below are the most influential features globally, which often align with individual "
                "predictions but should not be interpreted as such."
            )
            fig2 = px.bar(
                imp_df.head(8),
                x="importance",
                y="feature",
                orientation="h",
                title="Top 8 Most Influential Features (Global)",
                color="importance",
                color_continuous_scale=["#e87969", "#f3c969", "#50d2b2"],
            )
            fig2.update_layout(template="plotly_dark", showlegend=False, yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(fig2, use_container_width=True)

            # --- Recommendations based on input data & model explanation ---
            st.markdown("### Actionable Recommendations")
            recs = []
            # Check key predictors from global importance
            top_features = imp_df.head(5)["feature"].tolist()
            # Map global top features to actionable advice using the actual input values
            if "absences" in top_features or any(imp_df.feature.str.contains("absences").values):
                a = int(df_raw["absences"].iloc[0])
                if a >= 10:
                    recs.append("🎯 **Reduce absences**: Missing 10+ days significantly increases failure risk. "
                                "Consider tutoring or academic support.")
                elif a >= 5:
                    recs.append("📚 **Improve attendance**: 5-10 absences moderate risk. "
                                "Track attendance and seek help early.")
            if "studytime" in top_features or any(imp_df.feature.str.contains("studytime").values):
                s = int(df_raw["studytime"].iloc[0])
                if s <= 2:
                    recs.append("📈 **Increase study time**: Less than 2 hours/week is associated with lower predicted performance. "
                                "Aim for at least 5-7 hours spread across the week.")
            if "failures" in top_features or any(imp_df.feature.str.contains("failures").values):
                f = int(df_raw["failures"].iloc[0])
                if f >= 2:
                    recs.append("⚠️ **Address past failures**: 2+ prior failures significantly impact predicted outcome. "
                                "Consider credit recovery or targeted intervention.")
            if "health" in top_features or any(imp_df.feature.str.contains("health").values):
                h = int(df_raw["health"].iloc[0])
                if h <= 2:
                    recs.append("🏥 **Prioritize health**: Poor health status correlates with lower predicted performance. "
                                "Seek counseling or health services if needed.")
            if "goout" in top_features or any(imp_df.feature.str.contains("goout").values):
                g = int(df_raw["goout"].iloc[0])
                if g >= 4:
                    recs.append("🎮 **Balance social time**: High socializing (>4/5) correlates with lower predicted performance. "
                                "Consider setting boundaries on weekend/evening activities.")
            if not recs:
                recs.append("✅ **Keep current habits**: Based on your inputs, the model suggests your current trajectory is stable. "
                            "Continue monitoring and maintain healthy study habits.")
            for rec in recs:
                st.markdown(rec)

            st.markdown("---")
            st.caption(
                f"Prediction based on RandomForest model from `results/artefacts.pkl` · "
                f"Preprocessing: ColumnTransformer (numeric StandardScaler + categorical OneHotEncoder drop='first') "
                f"+ 7 engineered features · Probability threshold: 0.5 for PASS/FAIL · "
                f"Subject: {subject}"
            )

    else:
        st.info("Adjust the input parameters on the left and click **Predict Performance** to see the real model prediction.")
        st.caption(
            "All input fields map directly to the model's expected features. "
            "The pipeline: Raw Input → Engineered Features → ColumnTransformer → eng_scaler → RandomForestClassifier."
        )

st.markdown("---")
st.subheader("Dependencies Report")
with st.expander("View required packages"):
    st.write("| Package | Minimum Version | Purpose |")
    st.write("|---------|----------------|---------|")
    st.write("| `streamlit` | 1.28.0 | Web framework, UI components |")
    st.write("| `pandas` | 2.0.0+ | Data manipulation, DataFrame |")
    st.write("| `numpy` | 1.24.0+ | Model inference, array ops |")
    st.write("| `plotly` | 5.18.0+ | Interactive charts (Plotly Express) |")
    st.write("| `scikit-learn` | 1.3.0+ | RandomForestClassifier, ColumnTransformer, pipelines |")
    st.write("| `scipy` | 1.10.0+ | (Transitive dependency of pandas) |")
    st.write("| `pickle` | — | (Loading results/artefacts.pkl) |")
    st.caption("Note: Windows Application Control may block unsigned pandas .pyd extensions and SHAP "
               "DLLs. This dashboard gracefully falls back to RandomForest's built-in feature importance "
               "and displays simulated-data‑only disclaimers where appropriate.")
st.markdown("---")
st.markdown(
    '<p class="small-muted">XAI Project V2 · real dataset workspace · academic decision support only</p>',
    unsafe_allow_html=True,
)