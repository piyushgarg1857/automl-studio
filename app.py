import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from evaluation_core import detect_problem_type, evaluate_models
from ml_engine import CLASSIFIERS, REGRESSORS, get_dataset_profile

st.set_page_config(page_title="AutoML Studio", page_icon="⚡", layout="wide")
st.title("⚡ AutoML Studio")
st.caption("Tabular AutoML · Leakage-safe evaluation · Model comparison")

uploaded = st.file_uploader("Upload CSV, Excel, or JSON dataset", type=["csv", "xlsx", "json"])
if uploaded is None:
    st.info("Upload a dataset to explore it and evaluate machine-learning models.")
    st.stop()

try:
    suffix = uploaded.name.rsplit(".", 1)[-1].lower()
    if suffix == "csv":
        df = pd.read_csv(uploaded)
    elif suffix == "xlsx":
        df = pd.read_excel(uploaded)
    else:
        df = pd.read_json(uploaded)
except Exception as exc:
    st.error(f"Could not read dataset: {exc}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Dataset must contain rows and at least two columns (features plus target).")
    st.stop()

with st.sidebar:
    st.header("Experiment settings")
    target = st.selectbox("Target column", df.columns.tolist())
    task_choice = st.selectbox(
        "Problem type",
        ["Auto-detect", "Classification", "Regression"],
        help="Auto-detection can be ambiguous for numeric targets with few unique values.",
    )
    drop_cols = st.multiselect(
        "Exclude feature columns",
        [col for col in df.columns if col != target],
    )
    cv_folds = st.slider("Cross-validation folds", min_value=2, max_value=10, value=5)
    test_size = st.slider("Holdout test fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

task = {"Auto-detect": "auto", "Classification": "classification", "Regression": "regression"}[task_choice]
try:
    detected_task = detect_problem_type(df[target], task=task)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

# ── Dataset Summary Metrics ───────────────────────────────────────────────────
missing_total = int(df.isnull().sum().sum())
dup_rows = int(df.duplicated().sum())
num_cols = len(df.select_dtypes(include=np.number).columns)
cat_cols = len(df.select_dtypes(include="object").columns)

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card"><div class="val">{df.shape[0]:,}</div><div class="lbl">Total Rows</div></div>
    <div class="metric-card"><div class="val">{df.shape[1]}</div><div class="lbl">Total Columns</div></div>
    <div class="metric-card"><div class="val">{num_cols}</div><div class="lbl">Numeric Features</div></div>
    <div class="metric-card"><div class="val">{cat_cols}</div><div class="lbl">Categorical Features</div></div>
    <div class="metric-card"><div class="val">{missing_total}</div><div class="lbl">Missing Values</div></div>
    <div class="metric-card"><div class="val">{dup_rows}</div><div class="lbl">Duplicate Rows</div></div>
</div>
""", unsafe_allow_html=True)

# ── Step 2: Target + Config ───────────────────────────────────────────────────
st.markdown('<div class="step-header"><span class="step-badge">2</span> Configure & Explore</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 2])
with col_left:
    target = st.selectbox("Target Column (Prediction Variable)", df.columns.tolist())
    drop_cols = st.multiselect("Drop Columns (Optional)", [c for c in df.columns if c != target],
                                help="Remove irrelevant or ID columns before training")
    problem_type = detect_problem_type(df[target])
    badge_color = "#38a169" if problem_type == "classification" else "#805ad5"
    st.markdown(f"""
    <div style="margin-top:0.5rem;">
        <span style="background:{badge_color}; color:white; padding:4px 14px; border-radius:20px; font-size:0.82rem; font-weight:600;">
            🔍 {problem_type.upper()}
        </span>
        <span style="color:#a0aec0; font-size:0.8rem; margin-left:8px;">Auto-detected problem type</span>
    </div>
    """, unsafe_allow_html=True)

# ── Exploration Tabs ──────────────────────────────────────────────────────────
with col_right:
    exp_tab1, exp_tab2, exp_tab3 = st.tabs(["📋 Data Preview", "📊 EDA", "🗂️ Column Profile"])

    with exp_tab1:
        st.dataframe(df.head(20), use_container_width=True, height=260)

    with exp_tab2:
        num_df = df.select_dtypes(include=np.number)
        eda_choice = st.radio("Chart Type", ["Distribution", "Correlation Heatmap", "Class Balance"], horizontal=True)

        if eda_choice == "Distribution":
            feat = st.selectbox("Select Feature", num_df.columns.tolist(), key="eda_feat")
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="#1a202c")
            ax.set_facecolor("#1a202c")
            ax.hist(df[feat].dropna(), bins=30, color="#3182ce", edgecolor="#1a202c", alpha=0.85)
            ax.set_title(f"Distribution — {feat}", color="#e2e8f0", fontsize=10)
            ax.tick_params(colors="#a0aec0")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        elif eda_choice == "Correlation Heatmap":
            if len(num_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(6, 4), facecolor="#1a202c")
                ax.set_facecolor("#1a202c")
                sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="Blues",
                            ax=ax, linewidths=0.5, annot_kws={"size": 7})
                ax.tick_params(colors="#a0aec0", labelsize=7)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("Not enough numeric columns.")

        else:
            fig, ax = plt.subplots(figsize=(6, 3), facecolor="#1a202c")
            ax.set_facecolor("#1a202c")
            vc = df[target].value_counts()
            bars = ax.bar(vc.index.astype(str), vc.values, color="#3182ce", edgecolor="#1a202c")
            ax.set_title(f"Class Balance — {target}", color="#e2e8f0", fontsize=10)
            ax.tick_params(colors="#a0aec0")
            for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
            for bar, val in zip(bars, vc.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(val), ha="center", color="#e2e8f0", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with exp_tab3:
        profile = get_dataset_profile(df)
        st.dataframe(profile, use_container_width=True, height=260)

# ── Step 3: Train ─────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-badge">3</span> Train & Evaluate Models</div>', unsafe_allow_html=True)

if missing_total > 0:
    st.markdown(f'<div class="warn-box">⚠️ Dataset contains <b>{missing_total}</b> missing values. They will be handled using the <b>{impute_strategy}</b> strategy selected in the sidebar.</div>', unsafe_allow_html=True)

run_btn = st.button("⚡ Run AutoML Evaluation", use_container_width=True, type="primary")

if run_btn:
    with st.spinner("Training and evaluating all models using 5-fold cross-validation..."):
        try:
            X, y, le = preprocess(df, target, impute_strategy, scaler_type, drop_cols)
            feature_names = [c for c in df.drop(columns=drop_cols + [target] if drop_cols else [target]).columns]
            results, trained_models = evaluate_models(X, y, problem_type)
            st.session_state["results"] = results
            st.session_state["trained_models"] = trained_models
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["le"] = le
            st.session_state["feature_names"] = feature_names
            st.session_state["problem_type"] = problem_type
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

if "automl_results" not in st.session_state:
    st.stop()

results       = st.session_state["results"]
trained_models = st.session_state["trained_models"]
X             = st.session_state["X"]
y             = st.session_state["y"]
le            = st.session_state["le"]
feature_names = st.session_state["feature_names"]
problem_type  = st.session_state["problem_type"]
sort_col      = "CV Mean F1" if problem_type == "classification" else "CV Mean R²"
best          = results.iloc[0]
best_name     = best["Model"]
best_model, X_test, y_test = trained_models[best_name]

st.success(f"✅ Evaluation complete — {len(results)} models trained and ranked.")

# ── Best Model Banner ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="best-banner">
    <h2>🏆 Recommended Model: {best_name}</h2>
    <p>Ranked #1 based on {sort_col} across 5-fold cross-validation. Green bar in the leaderboard below.</p>
</div>
""", unsafe_allow_html=True)

metric_items = [(k, v) for k, v in best.items() if k != "Model"]
mcols = st.columns(len(metric_items))
for i, (k, v) in enumerate(metric_items):
    mcols[i].metric(k, v)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Result Tabs ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Leaderboard",
    "📈 Performance Chart",
    "🔍 Feature Importance",
    "🧩 Confusion Matrix",
    "🎛️ Hyperparameter Tuning"
])

# ── Tab 1: Leaderboard ────────────────────────────────────────────────────────
with tab1:
    def highlight_best(s):
        return ["background-color: #1a3a2a; color: #68d391; font-weight: bold"
                if i == 0 else "" for i in range(len(s))]
    st.dataframe(results.style.apply(highlight_best, axis=0), use_container_width=True)

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export Leaderboard as CSV", data=csv_bytes,
                       file_name="automl_leaderboard.csv", mime="text/csv")

# ── Tab 2: Performance Chart ──────────────────────────────────────────────────
with tab2:
    chart_metric = st.selectbox("Select Metric to Visualize", [c for c in results.columns if c != "Model"])
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#1a202c")
    ax.set_facecolor("#1a202c")
    colors = ["#48bb78" if i == 0 else "#3182ce" for i in range(len(results))]
    bars = ax.barh(results["Model"], results[chart_metric], color=colors, edgecolor="#1a202c", height=0.6)
    for bar, val in zip(bars, results[chart_metric]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", color="#e2e8f0", fontsize=8)
    ax.set_xlabel(chart_metric, color="#a0aec0")
    ax.set_title(f"Model Comparison — {chart_metric}", color="#e2e8f0", fontsize=12, pad=12)
    ax.tick_params(colors="#a0aec0")
    ax.invert_yaxis()
    for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

# ── Tab 3: Feature Importance ─────────────────────────────────────────────────
with tab3:
    fi_model_name = st.selectbox("Select Model", list(trained_models.keys()), index=0, key="fi_model")
    fi_model = trained_models[fi_model_name][0]
    fi_df = get_feature_importance(fi_model, feature_names)

    if fi_df is not None:
        max_features = len(fi_df)
        default_n = min(15, max_features)
        if max_features > 1:
            top_n = st.slider("Top N Features", 1, max_features, default_n)
        else:
            top_n = max_features
        fi_df = fi_df.head(top_n)
        fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)), facecolor="#1a202c")
        ax.set_facecolor("#1a202c")
        palette = sns.color_palette("Blues_r", len(fi_df))
        ax.barh(fi_df["Feature"], fi_df["Importance"], color=palette, edgecolor="#1a202c")
        ax.set_xlabel("Importance Score", color="#a0aec0")
        ax.set_title(f"Feature Importance — {fi_model_name}", color="#e2e8f0", fontsize=11, pad=10)
        ax.tick_params(colors="#a0aec0", labelsize=8)
        ax.invert_yaxis()
        for spine in ax.spines.values(): spine.set_edgecolor("#2d3748")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.dataframe(fi_df, use_container_width=True)
    else:
        st.info("This estimator does not expose built-in feature importance or coefficients.")

with matrix_tab:
    if task != "classification":
        st.info("Confusion matrices are available for classification tasks only.")
    else:
        cm_model_name = st.selectbox("Select Model", list(trained_models.keys()), index=0, key="cm_model")
        cm_model, cm_X_test, cm_y_test = trained_models[cm_model_name]
        cm, labels = get_confusion_matrix(cm_model, cm_X_test, cm_y_test, le)

        fig, ax = plt.subplots(figsize=(min(10, len(labels) + 3), min(8, len(labels) + 2)), facecolor="#1a202c")
        ax.set_facecolor("#1a202c")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, annot_kws={"size": 10})
        ax.set_xlabel("Predicted Label", color="#a0aec0", labelpad=10)
        ax.set_ylabel("True Label", color="#a0aec0", labelpad=10)
        ax.set_title(f"Confusion Matrix — {cm_model_name}", color="#e2e8f0", fontsize=12, pad=12)
        ax.tick_params(colors="#a0aec0")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ── Tab 5: Hyperparameter Tuning ──────────────────────────────────────────────
with tab5:
    st.markdown('<div class="info-box">🎛️ Runs RandomizedSearchCV (10 iterations, 3-fold CV) on the selected model to find optimal hyperparameters.</div>', unsafe_allow_html=True)

    TUNABLE = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM", "Decision Tree", "Extra Trees"]
    available_tunable = [m for m in TUNABLE if m in trained_models]

    if not available_tunable:
        st.warning("No tunable models available in results.")
    else:
        tune_model_name = st.selectbox("Select Model to Tune", available_tunable)
        tune_btn = st.button("🔧 Start Hyperparameter Tuning", use_container_width=True)

        if tune_btn:
            with st.spinner(f"Tuning {tune_model_name}... this may take a moment."):
                try:
                    base_model = trained_models[tune_model_name][0]
                    tuned_model, best_params = tune_best_model(tune_model_name, base_model, X, y, problem_type)

                    st.success(f"✅ Tuning complete for **{tune_model_name}**")
                    st.markdown("**Best Hyperparameters Found:**")
                    st.json(best_params)

                    # Evaluate tuned model
                    from sklearn.model_selection import train_test_split
                    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42,
                        stratify=y if problem_type == "classification" else None)
                    tuned_model.fit(X_tr, y_tr)
                    preds = tuned_model.predict(X_te)

                    if problem_type == "classification":
                        avg = "binary" if len(np.unique(y)) == 2 else "weighted"
                        t_acc = round(accuracy_score(y_te, preds), 4)
                        t_f1  = round(f1_score(y_te, preds, average=avg, zero_division=0), 4)
                        orig_acc = results[results["Model"] == tune_model_name]["Accuracy"].values[0]
                        orig_f1  = results[results["Model"] == tune_model_name]["F1-Score"].values[0]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Tuned Accuracy", t_acc, delta=round(t_acc - orig_acc, 4))
                        c2.metric("Tuned F1-Score", t_f1,  delta=round(t_f1  - orig_f1,  4))
                        c3.metric("Baseline Accuracy", orig_acc)
                        c4.metric("Baseline F1-Score", orig_f1)
                    else:
                        t_r2   = round(r2_score(y_te, preds), 4)
                        t_rmse = round(np.sqrt(mean_squared_error(y_te, preds)), 4)
                        orig_r2   = results[results["Model"] == tune_model_name]["R² Score"].values[0]
                        orig_rmse = results[results["Model"] == tune_model_name]["RMSE"].values[0]
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Tuned R² Score", t_r2,   delta=round(t_r2   - orig_r2,   4))
                        c2.metric("Tuned RMSE",     t_rmse, delta=round(t_rmse - orig_rmse, 4))
                        c3.metric("Baseline R²",    orig_r2)
                        c4.metric("Baseline RMSE",  orig_rmse)

                    # Export tuned model
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    model_bytes = export_model(tuned_model)
                    st.download_button(
                        f"💾 Download Tuned {tune_model_name} (.pkl)",
                        data=model_bytes,
                        file_name=f"{tune_model_name.lower().replace(' ', '_')}_tuned.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Tuning failed: {e}")

# ── Export Best Model ─────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="step-header"><span class="step-badge">4</span> Export Best Model</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    model_bytes = export_model(best_model)
    st.download_button(
        "Download best fitted pipeline (.pkl)",
        data=model_bytes,
        file_name=f"{best_name.lower().replace(' ', '_')}_pipeline.pkl",
        mime="application/octet-stream",
    )
