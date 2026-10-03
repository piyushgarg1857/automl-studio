# Development log

This file records meaningful development work, validation, known gaps, and handoff notes.

## 2026-10-03 — Export label handling, tests, and project handoff

### Scope
- Ensure exported classification models return original class labels rather than encoded integers.
- Add regression coverage for the exported model wrapper.
- Add a repeatable independent test command and maintain project documentation/checklists.

### Changes in this work
- Added `model_bundle.py` with a pickle-friendly wrapper around the fitted preprocessing/model pipeline.
- Updated the app's model export to include the target label encoder when one exists.
- Added tests for original-label prediction and serialization round-trip.
- Added/updated development checklist and AI handoff context.

### Validation record
- Syntax checks and unit tests must be run against the same branch/environment before merge.
- Do not mark tests as passing based only on code inspection.
- Record the exact command, environment, result, and any skipped checks below after running them.

### Current known gaps
- Hyperparameter tuning UI is not integrated into the current Streamlit app.
- Existing legacy evaluation/preprocessing functions remain in `ml_engine.py`; the app currently uses `evaluation_core.py`.
- The app's complete browser/UI flow and hosted deployment are not covered by the unit tests.

### Test runs
| Date | Command | Result | Notes |
|---|---|---|---|
| 2026-10-03 | Targeted ModelBundle tests (2 tests) | Passed | Independently executed in Python 3.13.5 with scikit-learn; full repository suite and app/UI runtime remain pending. |


## 2026-10-03 — Leakage-safe hyperparameter tuning UI

### Changes
- Added `tune_estimator()` to `evaluation_core.py`; tuning uses a pipeline with fold-local preprocessing and only the training split.
- Added a Tuning tab to the Streamlit app, with model selection, search-space display, randomized iteration control, best CV score/parameters, and tuned-model export.
- Added a focused test for the tuning core.
- Confirmed a GitHub Actions unittest workflow is present on the branch.

### Validation
- Pending: run the full repository suite and inspect the GitHub Actions result for this branch/PR.
- Manual Streamlit interaction and hosted deployment checks remain pending.


## 2026-10-03 — Evaluation regression and edge-case tests

### Changes
- Added tests for regression metrics and returned artifacts.
- Added a mixed-type dataset test with missing numeric and categorical feature values.
- Added a mismatched-row-count validation test.
- Updated the development checklist to distinguish completed coverage from remaining model-failure tests.

### Validation
- GitHub Actions is expected to run the full unittest suite for this branch/PR.
- Record the final run result after the workflow completes; no test result is assumed from code inspection.
