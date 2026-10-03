# Development checklist

Use this checklist to track project completion. Update it as work is completed; do not mark a task done without verification.

## Core app
- [x] Load CSV, Excel, and JSON files.
- [x] Preview dataset and show basic column profile.
- [x] Select target, problem type, excluded features, CV folds, and holdout fraction.
- [x] Evaluate model candidates and show a leaderboard.
- [x] Show metric chart, feature importance where supported, and classification confusion matrix.
- [x] Download leaderboard.
- [ ] Add user-facing validation for invalid/empty feature selections and small datasets.
- [ ] Add clear progress/status reporting for individual model failures.

## Evaluation correctness
- [x] Keep imputation, encoding, and scaling inside the sklearn pipeline.
- [x] Use stratified CV for classification and KFold for regression.
- [x] Encode classification targets for estimators that require integer classes.
- [ ] Add broader tests for missing values, multiclass edge cases, and small class counts.
- [ ] Add tests for regression metrics and model-failure handling.
- [ ] Review whether target encoding and class handling remain compatible with all supported estimators.

## Model export
- [x] Bundle fitted pipeline with target label encoder for classification export.
- [ ] Verify exported artifact in a clean environment using a saved sample input.
- [ ] Document safe loading and prediction usage for exported artifacts.

## Model tuning
- [ ] Integrate the existing tuning function and grids into the Streamlit UI.
- [ ] Show selected model, search space, scoring metric, and best parameters.
- [ ] Keep tuning preprocessing inside the pipeline to avoid leakage.
- [ ] Add tests for tuning behavior and runtime limits.

## Quality and delivery
- [ ] Run `python -m unittest discover -s tests -v` before each merge.
- [ ] Add CI workflow to run tests automatically on pushes and pull requests.
- [ ] Test the Streamlit app manually with classification and regression datasets.
- [ ] Verify deployment configuration and hosted app.
- [ ] Update `DEVELOPMENT_LOG.md` and `AI_CONTEXT.md` after substantial changes.
