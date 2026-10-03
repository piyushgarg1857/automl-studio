# AI / developer handoff context

Read this file together with `README.md`, `DEVELOPMENT_CHECKLIST.md`, and `DEVELOPMENT_LOG.md` before making changes.

## Project

AutoML Studio is a Streamlit-based tabular machine-learning application. It supports CSV, Excel, and JSON uploads; dataset profiling; target/task configuration; classification and regression model evaluation; visual diagnostics; randomized hyperparameter tuning; model export; prediction playground interactions; and saved experiment history.

## Current architecture

- `app.py`: Streamlit interface and workflow orchestration, including data-upload flow, evaluation results, tuning, prediction, exports, and experiment-history UI.
- `evaluation_core.py`: task detection, mixed-type preprocessing, leakage-safe evaluation, CV/holdout metrics, model artifacts/error capture, and randomized tuning.
- `experiment_store.py`: SQLite persistence helpers for saved experiment records.
- `model_bundle.py`: serializable wrapper that maps classification predictions back to original labels.
- `ml_engine.py`: model banks and older/legacy preprocessing and evaluation helpers. The development log notes the app currently uses `evaluation_core.py`; avoid expanding duplicated logic without a deliberate refactor.
- `tests/`: unittest coverage for evaluation, robustness/edge cases, model bundles, and tuning.
- `.github/workflows/tests.yml`: CI test workflow.
- `README.md`: product overview, architecture, flowcharts, setup, and known gaps.
- `DEVELOPMENT_CHECKLIST.md`: delivered features and remaining verification/work items.
- `DEVELOPMENT_LOG.md`: dated implementation and validation record.

## Delivered feature milestones

- **#10 — Feature Importance & Explainability:** feature-importance visualization where supported.
- **#11 — Prediction Playground:** interactive prediction flow for a fitted model.
- **#12 — Model Persistence & Experiment History:** SQLite-backed run history with load/delete/compare and leaderboard; the history screen is intended to remain accessible without a dataset upload. User reported testing complete on 2026-10-03.
- **#13 — Downloadable Model Evaluation Report:** proposed next feature; not implemented yet.

Use the checklist and log as the status source of truth. Do not infer that every environment or edge case is verified just because a feature exists.

## Important implementation decisions

- Keep learned imputation, encoding, and scaling inside sklearn pipelines to avoid preprocessing leakage across CV folds.
- Classification targets are encoded for estimator compatibility; user-facing predictions/diagnostics should preserve original labels.
- Evaluation uses stratified CV for classification and KFold for regression.
- Tuning should use only the training partition; do not pass the holdout set into randomized search.
- Model-level failures should be captured so one candidate does not terminate all evaluation.
- Treat SQLite on a hosted ephemeral filesystem as non-durable unless persistent storage is explicitly configured.
- Only load pickle/joblib artifacts from trusted sources.
- Never claim a test passed unless it was actually executed. Record command, environment, result, and skipped checks.

## Current priorities

1. Improve UI validation for invalid/empty feature selections and small datasets.
2. Make model-level progress and failure states clearer in the UI.
3. Expand edge-case tests, including missing values, multiclass behavior, and small class counts.
4. Verify exported artifacts in a clean environment and document safe prediction usage.
5. Add tuning runtime-limit tests.
6. Verify hosted deployment and persistence behavior.
7. Implement the evaluation report as a focused feature branch with tests and a reviewable PR.

## Development workflow

1. Inspect the current `datamining` branch and recent commits.
2. Create a focused feature branch; avoid direct base-branch edits for feature work.
3. Keep changes small and add/update tests.
4. Run `python -m unittest discover -s tests -v` and relevant syntax/UI checks.
5. Record exact validation results in `DEVELOPMENT_LOG.md`.
6. Update `DEVELOPMENT_CHECKLIST.md` only when completion is verified.
7. Open a PR with scope, test evidence, and remaining risks.
