import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance

from evaluation_core import detect_problem_type, evaluate_models, tune_estimator
from ml_engine import CLASSIFIERS, REGRESSORS, TUNING_GRIDS, get_dataset_profile
from model_bundle import ModelBundle
from experiment_store import save_experiment, list_experiments, load_experiment, delete_experiment

st.set_page_config(page_title="AutoML Studio", page_icon="⚡", layout="wide")

# Optional presentation layer: isolated from the modeling/evaluation logic.
st.markdown("""
<style>
:root { --bg:#061329; --panel:#0b1d3a; --panel2:#10264b; --ink:#eaf1ff; --muted:#9bb2d9; --line:#1d3b70; --blue:#3678ff; --violet:#6845f5; --cyan:#36c7e8; }
[data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 12% 0%, #142d66 0%, #081832 38%, #050f22 100%); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(5,15,34,.9); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#081832,#0b1c3b); border-right:1px solid #1b3970; }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1600px; }
h1,h2,h3,h4 { color:var(--ink); letter-spacing:-.025em; }
p,label,[data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stMetric"] { background:linear-gradient(145deg,#102852,#0b1c3a); border:1px solid #214783; border-radius:15px; padding:16px 18px; box-shadow:0 8px 28px rgba(0,0,0,.18); }
[data-testid="stMetricLabel"] { color:#b5c8eb; font-size:.84rem; }
[data-testid="stMetricValue"] { color:#f3f7ff; font-weight:750; }
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"] button { border-radius:10px; font-weight:650; min-height:2.65rem; border:1px solid #315aa1; background:#102750; color:#eaf1ff; transition:all .16s ease; }
.stButton>button:hover,.stDownloadButton>button:hover { border-color:#6e69ff; box-shadow:0 0 18px rgba(82,101,255,.25); transform:translateY(-1px); color:white; }
button[kind="primary"],[data-testid="stFormSubmitButton"] button[kind="primary"] { background:linear-gradient(110deg,#2876ff,#6246ee); border-color:#4b65ff; color:white; }
[data-testid="stTabs"] [role="tab"] { color:#a9bee4; font-weight:600; padding:.7rem .9rem; }
[data-testid="stTabs"] [aria-selected="true"] { color:#fff; border-bottom-color:#6482ff; }
[data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid #234477; border-radius:12px; overflow:hidden; }
[data-testid="stExpander"] { background:#0b1d3a; border:1px solid #214477; border-radius:12px; }
[data-testid="stFileUploader"] section { background:#0b1d3a; border:1.5px dashed #4265a6; border-radius:14px; }
[data-testid="stAlert"] { background:#10264a; border-color:#31558d; color:#eaf1ff; }
[data-testid="stSelectbox"]>div>div,[data-testid="stMultiSelect"]>div>div { background:#0b1d3a; }
hr { border-color:#1c3768; }
@media(max-width:768px) { .block-container {padding:1rem 1rem 2rem;} [data-testid="stMetric"]{padding:12px;} }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin:0 0 1rem 0;">
  <div style="width:52px;height:52px;border-radius:16px;background:linear-gradient(140deg,#36c7e8 0%,#3976ff 48%,#7546f5 100%);display:flex;align-items:center;justify-content:center;box-shadow:0 0 26px rgba(75,100,255,.35);">
    <svg width="34" height="34" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5 31L17 7H23L35 31H27L20 16L13 31H5Z" fill="white"/>
      <path d="M16 25H25L29 32H12L16 25Z" fill="#B9F5FF"/>
    </svg>
  </div>
  <div>
    <div style="font-size:1.8rem;line-height:1.15;font-weight:780;color:#F3F7FF;letter-spacing:-.04em;">AutoML <span style="background:linear-gradient(90deg,#58c9ff,#9b83ff);-webkit-background-clip:text;color:transparent;">Studio</span></div>
    <div style="font-size:.88rem;color:#9bb2d9;margin-top:5px;">Build · Train · Evaluate · Predict</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption("Leakage-safe evaluation  ·  Model comparison  ·  Explainability  ·  Experiment tracking")

def render_standalone_history():
    """Show saved runs even when no dataset is uploaded."""
    st.subheader("Saved experiment history")
    st.caption(
        "Your saved runs are stored in the local SQLite database. "
        "You can review, load, compare, or delete them without uploading a dataset."
    )
    try:
        history = list_experiments()
    except Exception as exc:
        st.error(f"Could not read experiment history: {type(exc).__name__}: {exc}")
        return

    if not history:
        st.info("No saved experiments yet. Upload a dataset, run an evaluation, and save it from the Experiment history tab.")
        return

    history_df = pd.DataFrame(history).rename(
        columns={
            "id": "ID", "name": "Experiment", "created_at": "Saved at (UTC)",
            "task": "Task", "target": "Target", "model_count": "Models",
            "score_metric": "Tracked metric", "score_value": "Tracked score",
        }
    )
    st.dataframe(history_df, use_container_width=True, hide_index=True)
    selected_id = st.selectbox(
        "Choose a saved experiment",
        [row["id"] for row in history],
        format_func=lambda value: next(
            f"#{row['id']} · {row['name']}" for row in history if row["id"] == value
        ),
        key="standalone_history_selection",
    )
    cols = st.columns(3)
    if cols[0].button("Load saved run", use_container_width=True, key="standalone_load"):
        try:
            payload = load_experiment(selected_id)
            st.session_state["automl_results"] = payload["leaderboard"]
            st.session_state["automl_artifacts"] = payload["artifacts"]
            meta = next(row for row in history if row["id"] == selected_id)
            st.session_state["automl_task"] = meta["task"]
            st.session_state["standalone_loaded_id"] = selected_id
            st.success("Saved leaderboard and fitted models loaded into this session.")
        except Exception as exc:
            st.error(f"Could not load experiment: {type(exc).__name__}: {exc}")
    if cols[1].button("Delete saved run", use_container_width=True, key="standalone_delete"):
        try:
            if delete_experiment(selected_id):
                st.success(f"Experiment #{selected_id} deleted.")
                st.rerun()
            else:
                st.warning("That experiment was not found.")
        except Exception as exc:
            st.error(f"Could not delete experiment: {type(exc).__name__}: {exc}")
    if cols[2].button("Compare saved runs", use_container_width=True, key="standalone_compare"):
        st.session_state["standalone_show_comparison"] = True

    loaded_id = st.session_state.get("standalone_loaded_id")
    if loaded_id is not None:
        st.markdown("#### Loaded experiment leaderboard")
        try:
            loaded = load_experiment(loaded_id)
            st.dataframe(loaded["leaderboard"], use_container_width=True, hide_index=True)
            st.caption("To use the loaded model in the prediction tools, upload a dataset to enter the full AutoML workspace.")
        except Exception as exc:
            st.warning(f"Could not display the loaded run: {exc}")

    if st.session_state.get("standalone_show_comparison"):
        st.markdown("#### Compare saved runs")
        st.dataframe(
            pd.DataFrame([
                {
                    "Experiment": row["name"], "Task": row["task"],
                    "Target": row["target"], "Metric": row["score_metric"],
                    "Score": row["score_value"], "Models": row["model_count"],
                }
                for row in history
            ]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption("Compare scores only when datasets, targets, tasks, and metrics are compatible.")


uploaded = st.file_uploader(
    "Upload CSV, Excel, or JSON dataset", type=["csv", "xlsx", "json"]
)
if uploaded is None:
    st.info("Upload a dataset to explore it and evaluate machine-learning models, or use saved history below.")
    render_standalone_history()
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
        "Exclude feature columns", [col for col in df.columns if col != target]
    )
    cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
    test_size = st.slider(
        "Holdout test fraction", min_value=0.1, max_value=0.4, value=0.2, step=0.05
    )

task = {
    "Auto-detect": "auto",
    "Classification": "classification",
    "Regression": "regression",
}[task_choice]
try:
    detected_task = detect_problem_type(df[target], task=task)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

summary = st.columns(5)
summary[0].metric("Rows", f"{len(df):,}")
summary[1].metric("Columns", len(df.columns))
summary[2].metric("Numeric", len(df.select_dtypes(include=np.number).columns))
summary[3].metric("Missing cells", int(df.isna().sum().sum()))
summary[4].metric("Duplicate rows", int(df.duplicated().sum()))
st.caption(f"Selected task: **{detected_task.title()}**")

with st.expander("Dataset preview and column profile", expanded=True):
    preview_tab, profile_tab = st.tabs(["Preview", "Column profile"])
    with preview_tab:
        st.dataframe(df.head(100), use_container_width=True)
    with profile_tab:
        st.dataframe(get_dataset_profile(df), use_container_width=True)

model_bank = CLASSIFIERS if detected_task == "classification" else REGRESSORS
st.subheader("Model selection")
evaluation_mode = st.radio(
    "Evaluation mode",
    ["All models", "Selected models"],
    horizontal=True,
    help="Run every available model or choose a custom subset for this experiment.",
)
if evaluation_mode == "Selected models":
    selected_model_names = st.multiselect(
        "Models to evaluate",
        options=list(model_bank.keys()),
        default=list(model_bank.keys()),
        help="Remove models you do not want to include. Select all to run the full model bank.",
    )
    models_to_run = {
        name: model_bank[name]
        for name in selected_model_names
    }
else:
    models_to_run = model_bank

st.caption(f"Ready to evaluate {len(models_to_run)} of {len(model_bank)} available models.")

if st.button(
    "Run model evaluation",
    type="primary",
    use_container_width=True,
    disabled=not models_to_run,
):
    X = df.drop(columns=[target] + drop_cols)
    y = df[target]
    progress_bar = st.progress(0, text="Preparing evaluation...")
    progress_text = st.empty()
    started_at = time.perf_counter()

    def update_evaluation_progress(event):
        fraction = event["index"] / event["total"]
        progress_bar.progress(fraction, text=f"Model {event['index']} of {event['total']}")
        if event["status"] == "started":
            progress_text.info(f"Evaluating {event['model']}...")
        else:
            progress_text.write(
                f"{event['model']}: {event['status']} · {event.get('elapsed', 0):.2f}s"
            )

    with st.spinner("Evaluating models. Preprocessing is fitted inside each CV fold..."):
        try:
            leaderboard, artifacts = evaluate_models(
                X,
                y,
                models_to_run,
                problem_type=detected_task,
                test_size=test_size,
                cv_folds=cv_folds,
                progress_callback=update_evaluation_progress,
            )
            progress_bar.progress(1.0, text="Evaluation complete")
            st.success(f"Evaluation finished in {time.perf_counter() - started_at:.2f} seconds.")
            st.session_state["automl_results"] = leaderboard
            st.session_state["automl_artifacts"] = artifacts
            st.session_state["automl_task"] = detected_task
        except Exception as exc:
            st.error(f"Evaluation could not start: {type(exc).__name__}: {exc}")
            st.stop()

if "automl_results" not in st.session_state:
    st.stop()

results = st.session_state["automl_results"]
artifacts = st.session_state["automl_artifacts"]
task = st.session_state["automl_task"]
score_col = "CV F1 Mean" if task == "classification" else "CV R² Mean"
valid = results[results[score_col].notna()] if score_col in results else pd.DataFrame()

if valid.empty:
    st.error("No model completed successfully. Expand the errors table below for details.")
    st.dataframe(results, use_container_width=True)
    st.stop()

best_row = valid.iloc[0]
best_name = best_row["Model"]
st.subheader("Evaluation results")
st.caption(
    f"Leaderboard is ordered by {score_col}. "
    "Holdout metrics are reported separately from CV."
)
metric_cols = st.columns(min(5, len([c for c in valid.columns if c != "Model"])))
shown_metrics = [c for c in valid.columns if c != "Model"][: len(metric_cols)]
for col, metric in zip(metric_cols, shown_metrics):
    value = best_row[metric]
    col.metric(metric, f"{value:.4f}" if pd.notna(value) else "—")

leader_tab, insights_tab, chart_tab, explain_tab, matrix_tab, predict_tab, history_tab, export_tab, tuning_tab = st.tabs(
    ["Leaderboard", "Model insights", "Comparison dashboard", "Feature importance", "Confusion matrix", "Prediction playground", "Experiment history", "Export", "Tuning"]
)

with leader_tab:
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.download_button(
        "Download leaderboard CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="automl_leaderboard.csv",
        mime="text/csv",
    )
    failures = results[results["Error"].notna()] if "Error" in results else pd.DataFrame()
    if not failures.empty:
        with st.expander(f"Model errors ({len(failures)})"):
            st.dataframe(failures[["Model", "Error"]], use_container_width=True)

with insights_tab:
    st.subheader("Automated evaluation insights")
    st.caption(
        "These observations summarize the current evaluation results. "
        "They are descriptive and should be considered alongside your data and use case."
    )
    if valid.empty:
        st.info("Run a successful model evaluation to generate insights.")
    else:
        score_metric = "CV F1 Mean" if task == "classification" else "CV R² Mean"
        error_metrics = ["RMSE", "MAE"]
        timed = valid[valid["Evaluation Time (s)"].notna()].copy()
        scored = valid[valid[score_metric].notna()].copy()
        if not scored.empty:
            top = scored.sort_values(score_metric, ascending=False).iloc[0]
            st.markdown(
                f"**Highest {score_metric}:** {top['Model']} "
                f"({top[score_metric]:.4f})."
            )
            spread = float(scored[score_metric].max() - scored[score_metric].min())
            st.write(
                f"**Score range:** {spread:.4f} across {len(scored)} successful model(s) "
                f"using {score_metric}."
            )
        if not timed.empty:
            fastest = timed.sort_values("Evaluation Time (s)").iloc[0]
            st.write(
                f"**Shortest evaluation time:** {fastest['Model']} "
                f"({fastest['Evaluation Time (s)']:.3f} seconds)."
            )
        available_errors = [m for m in error_metrics if m in valid.columns]
        if task == "regression" and available_errors:
            for metric in available_errors:
                best_error = valid[valid[metric].notna()].sort_values(metric).head(1)
                if not best_error.empty:
                    row = best_error.iloc[0]
                    st.write(f"**Lowest {metric}:** {row['Model']} ({row[metric]:.4f}).")
        if "Error" in results:
            failed = results[results["Error"].notna()]
            if not failed.empty:
                st.warning(
                    f"{len(failed)} model(s) failed. Expand the model errors section "
                    "in the Leaderboard tab to inspect details."
                )
        st.info(
            "Tip: Compare cross-validation scores with holdout metrics. "
            "A small score difference does not by itself establish that one model "
            "will perform better on future data."
        )

with chart_tab:
    st.subheader("Compare model performance")
    numeric_metrics = [
        c for c in valid.select_dtypes(include=np.number).columns if c != "Model"
    ]
    if numeric_metrics:
        model_options = valid["Model"].tolist()
        chosen_models = st.multiselect(
            "Models to compare",
            options=model_options,
            default=model_options,
            key="comparison_models",
        )
        default_metrics = [
            metric for metric in numeric_metrics
            if metric not in {"CV F1 Std", "CV R² Std"}
        ]
        chosen_metrics = st.multiselect(
            "Metrics to visualize",
            options=numeric_metrics,
            default=default_metrics,
            key="comparison_metrics",
            help="Metrics use separate charts because their scales and meanings differ.",
        )
        comparison = valid[valid["Model"].isin(chosen_models)].copy()
        if comparison.empty:
            st.info("Select at least one model to display the comparison.")
        else:
            st.caption(
                "The table and charts show the selected models only. "
                "Lower values are preferable for error and time metrics; "
                "higher values are preferable for scores."
            )
            display_cols = ["Model"] + chosen_metrics
            st.dataframe(
                comparison[display_cols],
                use_container_width=True,
                hide_index=True,
            )
            for start in range(0, len(chosen_metrics), 2):
                chart_cols = st.columns(min(2, len(chosen_metrics) - start))
                for offset, metric in enumerate(chosen_metrics[start:start + 2]):
                    chart = comparison.sort_values(
                        metric,
                        ascending=metric in {"RMSE", "MAE", "Evaluation Time (s)"},
                    )
                    fig, ax = plt.subplots(
                        figsize=(8, max(3.5, len(chart) * 0.32))
                    )
                    ax.barh(chart["Model"], chart[metric])
                    ax.set_xlabel(metric)
                    ax.set_title(metric)
                    ax.invert_yaxis()
                    fig.tight_layout()
                    chart_cols[offset].pyplot(fig, use_container_width=True)
                    plt.close(fig)
    else:
        st.info("No numeric metrics are available to compare.")

with explain_tab:
    st.subheader("Feature importance and explainability")
    st.caption(
        "Explore which input features are associated with a fitted model's predictions. "
        "Importance is model- and dataset-dependent; it does not establish causation."
    )
    successful = [
        name for name, item in artifacts.items() if item.get("pipeline") is not None
    ]
    if successful:
        selected = st.selectbox("Model", successful, key="importance_model")
        method = st.radio(
            "Importance method",
            ["Permutation importance", "Built-in importance / coefficients"],
            horizontal=True,
            key="importance_method",
            help=(
                "Permutation importance works with any fitted estimator. Built-in "
                "importance is available only for estimators that expose it."
            ),
        )
        pipeline = artifacts[selected]["pipeline"]
        if method == "Built-in importance / coefficients":
            estimator = pipeline.named_steps["model"]
            preprocessor = pipeline.named_steps["preprocess"]
            importance = getattr(estimator, "feature_importances_", None)
            coefficients = getattr(estimator, "coef_", None)
            if importance is not None or coefficients is not None:
                names = preprocessor.get_feature_names_out()
                values = np.asarray(
                    importance if importance is not None else coefficients
                )
                if values.ndim > 1:
                    values = np.mean(np.abs(values), axis=0)
                if len(names) == len(values):
                    fi = pd.DataFrame(
                        {"Feature": names, "Importance": np.abs(values)}
                    ).sort_values("Importance", ascending=False).head(25)
                    st.dataframe(fi, use_container_width=True, hide_index=True)
                    fig, ax = plt.subplots(figsize=(9, max(4, len(fi) * 0.3)))
                    ax.barh(fi["Feature"], fi["Importance"])
                    ax.set_xlabel("Importance (absolute magnitude)")
                    ax.invert_yaxis()
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info(
                        "Feature names could not be aligned with this model's importance values."
                    )
            else:
                st.info(
                    "This estimator does not expose built-in importance or coefficients. "
                    "Choose permutation importance to explain it."
                )
        else:
            item = artifacts[selected]
            st.write(
                "Permutation importance measures how much the selected score changes "
                "when one input column is shuffled. Larger drops indicate greater "
                "reliance on that column for this evaluation dataset."
            )
            repeats = st.slider(
                "Permutation repeats", min_value=3, max_value=15, value=5,
                key="permutation_repeats",
                help="More repeats make the estimate less noisy but take longer.",
            )
            if st.button(
                "Calculate permutation importance",
                key="calculate_permutation_importance",
            ):
                y_for_scoring = item["y_test"]
                if item.get("label_encoder") is not None:
                    y_for_scoring = item["label_encoder"].transform(
                        np.asarray(y_for_scoring)
                    )
                scoring = "f1_weighted" if task == "classification" else "r2"
                with st.spinner("Calculating permutation importance..."):
                    try:
                        result = permutation_importance(
                            pipeline,
                            item["X_test"],
                            y_for_scoring,
                            scoring=scoring,
                            n_repeats=repeats,
                            random_state=42,
                            n_jobs=-1,
                        )
                        fi = pd.DataFrame(
                            {
                                "Feature": item["X_test"].columns,
                                "Importance Mean": result.importances_mean,
                                "Importance Std": result.importances_std,
                            }
                        ).sort_values("Importance Mean", ascending=False)
                        st.session_state["permutation_importance_result"] = {
                            "model": selected,
                            "data": fi,
                        }
                    except Exception as exc:
                        st.error(
                            f"Could not calculate permutation importance: "
                            f"{type(exc).__name__}: {exc}"
                        )
            saved = st.session_state.get("permutation_importance_result")
            if saved and saved["model"] == selected:
                fi = saved["data"].head(25)
                st.dataframe(fi, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(9, max(4, len(fi) * 0.32)))
                ax.barh(
                    fi["Feature"],
                    fi["Importance Mean"],
                    xerr=fi["Importance Std"],
                )
                ax.axvline(0, linewidth=1)
                ax.set_xlabel(f"Mean decrease in {('weighted F1' if task == 'classification' else 'R²')}")
                ax.invert_yaxis()
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                st.caption(
                    "Error bars show the standard deviation across shuffles. "
                    "Negative or near-zero values can occur and should be interpreted cautiously."
                )
            st.warning(
                "Permutation importance is calculated on the holdout set. Use it to "
                "understand this evaluation, not to repeatedly tune choices against "
                "the holdout and then report that same score as an unbiased final estimate."
            )
    else:
        st.info("No model completed successfully.")

with matrix_tab:
    if task != "classification":
        st.info("Confusion matrices are available for classification tasks only.")
    else:
        successful = [
            name for name, item in artifacts.items() if item.get("pipeline") is not None
        ]
        if successful:
            selected = st.selectbox("Model", successful, key="matrix_model")
            item = artifacts[selected]
            labels = np.unique(
                np.concatenate(
                    [np.asarray(item["y_test"]), np.asarray(item["predictions"])]
                )
            )
            cm = confusion_matrix(
                item["y_test"], item["predictions"], labels=labels
            )
            fig, ax = plt.subplots(
                figsize=(max(5, len(labels) * 0.7), max(4, len(labels) * 0.6))
            )
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No classification model completed successfully.")


with predict_tab:
    st.subheader("Prediction playground")
    st.caption(
        "Generate predictions from a fitted model using new input values. "
        "Inputs are passed through the same fitted preprocessing pipeline."
    )
    predict_models = [
        name for name, item in artifacts.items() if item.get("pipeline") is not None
    ]
    if not predict_models:
        st.info("Run a successful model evaluation before generating predictions.")
    else:
        prediction_model = st.selectbox(
            "Model for prediction", predict_models, key="playground_model"
        )
        prediction_artifact = artifacts[prediction_model]
        reference_X = prediction_artifact["X_test"]
        st.markdown("#### Single prediction")
        input_values = {}
        with st.form("single_prediction_form"):
            input_cols = st.columns(2)
            for idx, feature in enumerate(reference_X.columns):
                series = reference_X[feature]
                with input_cols[idx % 2]:
                    if pd.api.types.is_numeric_dtype(series):
                        numeric_values = pd.to_numeric(series, errors="coerce").dropna()
                        default = float(numeric_values.median()) if not numeric_values.empty else 0.0
                        input_values[feature] = st.number_input(
                            str(feature), value=default, key=f"pred_{feature}"
                        )
                    else:
                        choices = series.dropna().astype(str).unique().tolist()
                        choices = choices[:200] if choices else ["Unknown"]
                        input_values[feature] = st.selectbox(
                            str(feature), choices, key=f"pred_{feature}"
                        )
            single_submit = st.form_submit_button("Generate prediction", type="primary")
        if single_submit:
            row = pd.DataFrame([input_values], columns=reference_X.columns)
            try:
                raw_prediction = prediction_artifact["pipeline"].predict(row)
                encoder = prediction_artifact.get("label_encoder")
                if task == "classification":
                    label = encoder.inverse_transform(np.asarray(raw_prediction, dtype=int))[0] if encoder is not None else raw_prediction[0]
                    st.success(f"Predicted class: {label}")
                    estimator = prediction_artifact["pipeline"].named_steps["model"]
                    if hasattr(estimator, "predict_proba"):
                        probabilities = prediction_artifact["pipeline"].predict_proba(row)[0]
                        classes = estimator.classes_
                        display_classes = encoder.inverse_transform(np.asarray(classes, dtype=int)) if encoder is not None else classes
                        st.dataframe(
                            pd.DataFrame({"Class": display_classes, "Probability": probabilities}).sort_values(
                                "Probability", ascending=False
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                else:
                    st.success(f"Predicted value: {float(np.asarray(raw_prediction).ravel()[0]):.6g}")
            except Exception as exc:
                st.error(f"Prediction failed: {type(exc).__name__}: {exc}")

        st.divider()
        st.markdown("#### Batch predictions")
        batch_file = st.file_uploader(
            "Upload CSV with feature columns", type=["csv"], key="batch_prediction_file"
        )
        if batch_file is not None:
            try:
                batch_X = pd.read_csv(batch_file)
                missing = [col for col in reference_X.columns if col not in batch_X.columns]
                extra = [col for col in batch_X.columns if col not in reference_X.columns]
                if missing:
                    st.error(f"Missing required feature columns: {', '.join(map(str, missing))}")
                elif extra:
                    st.warning(f"Extra columns will be ignored: {', '.join(map(str, extra))}")
                    batch_X = batch_X[reference_X.columns]
                else:
                    batch_X = batch_X[reference_X.columns]
                if not missing:
                    batch_predictions = prediction_artifact["pipeline"].predict(batch_X)
                    encoder = prediction_artifact.get("label_encoder")
                    if task == "classification" and encoder is not None:
                        batch_predictions = encoder.inverse_transform(
                            np.asarray(batch_predictions, dtype=int)
                        )
                    output = batch_X.copy()
                    output["Prediction"] = batch_predictions
                    st.dataframe(output.head(100), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download predictions CSV",
                        output.to_csv(index=False).encode("utf-8"),
                        file_name="automl_predictions.csv",
                        mime="text/csv",
                        key="download_batch_predictions",
                    )
            except Exception as exc:
                st.error(f"Could not process batch file: {type(exc).__name__}: {exc}")


with history_tab:
    st.subheader("Experiment history")
    st.caption(
        "Save this run locally in SQLite, revisit its metrics, and download fitted models. "
        "The database stays on the machine running this Streamlit app."
    )
    with st.form("save_experiment_form"):
        experiment_name = st.text_input(
            "Experiment name",
            value=f"{uploaded.name.rsplit('.', 1)[0]} · {task.title()}",
            max_chars=100,
        )
        save_clicked = st.form_submit_button("Save current experiment", type="primary")
    if save_clicked:
        if not experiment_name.strip():
            st.error("Enter a name for this experiment.")
        else:
            try:
                experiment_id = save_experiment(
                    experiment_name.strip(), task, target, results, artifacts, score_col
                )
                st.success(f"Saved experiment #{experiment_id}.")
            except Exception as exc:
                st.error(f"Could not save experiment: {type(exc).__name__}: {exc}")

    try:
        history = list_experiments()
    except Exception as exc:
        history = []
        st.error(f"Could not read experiment history: {type(exc).__name__}: {exc}")

    if history:
        history_df = pd.DataFrame(history).rename(
            columns={
                "id": "ID", "name": "Experiment", "created_at": "Saved at (UTC)",
                "task": "Task", "target": "Target", "model_count": "Models",
                "score_metric": "Tracked metric", "score_value": "Tracked score",
            }
        )
        st.dataframe(history_df, use_container_width=True, hide_index=True)
        id_options = [row["id"] for row in history]
        selected_history_id = st.selectbox(
            "Choose a saved experiment",
            id_options,
            format_func=lambda value: next(
                f"#{row['id']} · {row['name']}" for row in history if row["id"] == value
            ),
            key="history_selection",
        )
        action_cols = st.columns(3)
        if action_cols[0].button("Load experiment", use_container_width=True):
            try:
                saved_payload = load_experiment(selected_history_id)
                st.session_state["automl_results"] = saved_payload["leaderboard"]
                st.session_state["automl_artifacts"] = saved_payload["artifacts"]
                selected_meta = next(row for row in history if row["id"] == selected_history_id)
                st.session_state["automl_task"] = selected_meta["task"]
                st.success("Saved results loaded into the current session. Re-upload the matching dataset if needed.")
            except Exception as exc:
                st.error(f"Could not load experiment: {type(exc).__name__}: {exc}")
        if action_cols[1].button("Delete experiment", use_container_width=True):
            try:
                if delete_experiment(selected_history_id):
                    st.success(f"Experiment #{selected_history_id} deleted.")
                    st.rerun()
                else:
                    st.warning("That experiment was not found.")
            except Exception as exc:
                st.error(f"Could not delete experiment: {type(exc).__name__}: {exc}")
        if action_cols[2].button("Compare saved runs", use_container_width=True):
            st.session_state["show_saved_comparison"] = True

        if st.session_state.get("show_saved_comparison"):
            st.markdown("#### Saved run comparison")
            compare_rows = [
                {
                    "Experiment": row["name"],
                    "Task": row["task"],
                    "Target": row["target"],
                    "Metric": row["score_metric"],
                    "Score": row["score_value"],
                    "Models": row["model_count"],
                }
                for row in history
            ]
            st.dataframe(pd.DataFrame(compare_rows), use_container_width=True, hide_index=True)
            st.caption("Runs may use different datasets, targets, or metrics. Compare scores only when those conditions are compatible.")
    else:
        st.info("No saved experiments yet. Save the current run to begin your history.")

with export_tab:
    st.write(f"Selected export model: **{best_name}**")
    st.warning("Pickle files should only be loaded from sources you trust.")
    model_bytes = pickle.dumps(
        ModelBundle(
            artifacts[best_name]["pipeline"],
            artifacts[best_name].get("label_encoder"),
        )
    )
    st.download_button(
        "Download best fitted pipeline (.pkl)",
        data=model_bytes,
        file_name=f"{best_name.lower().replace(' ', '_')}_pipeline.pkl",
        mime="application/octet-stream",
    )


with tuning_tab:
    st.subheader("Hyperparameter tuning")
    st.caption(
        "Search runs only on the training split. The holdout test data is not used "
        "to choose parameters, and preprocessing is fitted inside each CV fold."
    )
    available = CLASSIFIERS if task == "classification" else REGRESSORS
    tunable_names = [name for name in TUNING_GRIDS if name in available]
    if not tunable_names:
        st.info("No tuning grids are configured for this problem type.")
    else:
        selected_tune = st.selectbox("Model to tune", tunable_names, key="tuning_model")
        st.write("Search space")
        st.json(TUNING_GRIDS[selected_tune])
        iterations = st.slider(
            "Randomized search iterations", min_value=1, max_value=30, value=10,
            key="tuning_iterations",
            help="More iterations may find better settings but take longer.",
        )
        if st.button("Start hyperparameter tuning", type="primary", key="start_tuning"):
            X_tune = df.drop(columns=[target] + drop_cols)
            y_tune = df[target]
            with st.spinner("Searching parameter combinations..."):
                try:
                    search, encoder, tuned_task = tune_estimator(
                        X_tune,
                        y_tune,
                        available[selected_tune],
                        TUNING_GRIDS[selected_tune],
                        problem_type=task,
                        test_size=test_size,
                        cv_folds=cv_folds,
                        n_iter=iterations,
                    )
                    st.session_state["tuning_result"] = {
                        "model": selected_tune,
                        "search": search,
                        "encoder": encoder,
                        "task": tuned_task,
                    }
                except Exception as exc:
                    st.error(f"Tuning failed: {type(exc).__name__}: {exc}")

        tuned = st.session_state.get("tuning_result")
        if tuned:
            st.markdown(f"**Last tuned model:** {tuned['model']}")
            st.metric("Best cross-validation score", f"{tuned['search'].best_score_:.4f}")
            st.write("Best parameters")
            st.json(tuned["search"].best_params_)
            tuned_bundle = ModelBundle(
                tuned["search"].best_estimator_,
                tuned.get("encoder"),
            )
            st.download_button(
                "Download tuned model (.pkl)",
                data=pickle.dumps(tuned_bundle),
                file_name=f"{tuned['model'].lower().replace(' ', '_')}_tuned.pkl",
                mime="application/octet-stream",
                key="download_tuned_model",
            )
