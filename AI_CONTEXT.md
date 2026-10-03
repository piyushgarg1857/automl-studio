# AI / developer handoff context

This file is a concise project context for a future developer or AI assistant. Read this before continuing development.

## Project
A Streamlit-based tabular AutoML application. It accepts CSV, Excel, and JSON datasets, profiles data, lets the user select a target and task, evaluates a bank of classification/regression models, compares metrics, and offers visualizations and model export.

## Architecture
- `app.py`: Streamlit UI and orchestration.
- `evaluation_core.py`: task detection, preprocessing pipeline construction, holdout evaluation, cross-validation, leaderboard, and per-model artifacts.
- `ml_engine.py`: model banks, dataset profile helper, legacy preprocessing/evaluation utilities, and tuning grids/function.
- `model_bundle.py`: serializable wrapper that decodes predicted class IDs back to original labels for exported models.
- `tests/`: unittest-based tests.
- `DEVELOPMENT_LOG.md`: chronological work and validation record.
- `DEVELOPMENT_CHECKLIST.md`: outstanding and completed project tasks.

## Important implementation decisions
- Learned feature preprocessing belongs inside sklearn pipelines so it is fitted only on training folds.
- Classification targets are label-encoded inside `evaluate_models`; metrics are computed on encoded values, which preserves classification metric meaning.
- For user-facing confusion matrices, artifacts restore original labels.
- Exported classification models should use `ModelBundle` so downstream `predict()` returns original labels. Regression predictions remain numeric.
- Never claim a test passed unless it was actually executed. Record environment and command.

## Current priorities
1. Validate `ModelBundle` export and prediction behavior with automated tests.
2. Run the complete unittest suite independently and record the actual outcome.
3. Integrate hyperparameter tuning into the UI without preprocessing leakage.
4. Reduce duplication by deciding whether to remove, deprecate, or reuse legacy evaluation code in `ml_engine.py`.
5. Add CI and broader edge-case tests.

## How to continue
1. Read `DEVELOPMENT_LOG.md` and `DEVELOPMENT_CHECKLIST.md`.
2. Inspect current branch and recent commits before editing.
3. Create a focused feature branch; avoid directly changing the base branch.
4. Make small changes and add tests.
5. Run `python -m unittest discover -s tests -v` and syntax checks.
6. Record test results and limitations in the development log.
7. Open a PR with summary, test evidence, and remaining risks.
