import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix

from evaluation_core import detect_problem_type, evaluate_models, tune_estimator
from ml_engine import CLASSIFIERS, REGRESSORS, TUNING_GRIDS, get_dataset_profile
from model_bundle import ModelBundle

st.set_page_config(page_title="AutoML Studio", page_icon="⚡", layout="wide")
st.title("⚡ AutoML Studio")
st.caption("Tabular AutoML · Leakage-safe evaluation · Model comparison")

uploaded = st.file_uploader(
    "Upload CSV, Excel, or JSON dataset", type=["csv", "xlsx", "json"]
)
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

leader_tab, insights_tab, chart_tab, explain_tab, matrix_tab, export_tab, tuning_tab = st.tabs(
    ["Leaderboard", "Model insights", "Comparison dashboard", "Feature importance", "Confusion matrix", "Export", "Tuning"]
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
    successful = [
        name for name, item in artifacts.items() if item.get("pipeline") is not None
    ]
    if successful:
        selected = st.selectbox("Model", successful, key="importance_model")
        pipeline = artifacts[selected]["pipeline"]
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
                "This estimator does not expose built-in feature importance or coefficients."
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
