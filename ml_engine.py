import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor,
                               AdaBoostClassifier, AdaBoostRegressor,
                               ExtraTreesClassifier, ExtraTreesRegressor)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, f1_score, mean_squared_error,
                              r2_score, confusion_matrix)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import warnings
warnings.filterwarnings("ignore")


def detect_problem_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.nunique() <= 20:
        return "classification"
    return "regression"


def get_dataset_profile(df: pd.DataFrame) -> pd.DataFrame:
    profile = []
    for col in df.columns:
        profile.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Missing": int(df[col].isnull().sum()),
            "Missing %": round(df[col].isnull().mean() * 100, 2),
            "Unique Values": int(df[col].nunique()),
            "Skewness": round(float(df[col].skew()), 3) if pd.api.types.is_numeric_dtype(df[col]) else float("nan"),
        })
    return pd.DataFrame(profile)


def preprocess(df: pd.DataFrame, target: str,
               impute_strategy: str = "mean",
               scaler_type: str = "standard",
               drop_cols: list = None):
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    X = df.drop(columns=[target])
    y = df[target].copy()

    le = None
    if y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target)

    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy=impute_strategy)
    X_imputed = imputer.fit_transform(X)

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        return X_imputed, np.array(y), le

    X_processed = scaler.fit_transform(X_imputed)
    return X_processed, np.array(y), le


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42, class_weight="balanced"),
    "Decision Tree":       DecisionTreeClassifier(random_state=42, class_weight="balanced"),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "AdaBoost":            AdaBoostClassifier(random_state=42),
    "Extra Trees":         ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "KNN":                 KNeighborsClassifier(),
    "SVM":                 SVC(probability=True, random_state=42, class_weight="balanced"),
    "Naive Bayes":         GaussianNB(),
    "XGBoost":             XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42),
    "LightGBM":            LGBMClassifier(verbose=-1, random_state=42, class_weight="balanced"),
}

REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Ridge":             Ridge(random_state=42),
    "Lasso":             Lasso(random_state=42),
    "Decision Tree":     DecisionTreeRegressor(random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost":          AdaBoostRegressor(random_state=42),
    "Extra Trees":       ExtraTreesRegressor(n_estimators=100, random_state=42),
    "KNN":               KNeighborsRegressor(),
    "SVR":               SVR(),
    "XGBoost":           XGBRegressor(verbosity=0, random_state=42),
    "LightGBM":          LGBMRegressor(verbose=-1, random_state=42),
}

TUNING_GRIDS = {
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10], "min_samples_split": [2, 5]},
    "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5]},
    "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "max_depth": [3, 5, 7]},
    "LightGBM": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
    "Decision Tree": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "Extra Trees": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]},
}


def evaluate_models(X, y, problem_type: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" else None
    )
    models = CLASSIFIERS if problem_type == "classification" else REGRESSORS
    results, trained_models = [], {}

    for name, model in models.items():
        try:
            cv_scoring = "f1_weighted" if problem_type == "classification" else "r2"
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=cv_scoring)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem_type == "classification":
                avg = "binary" if len(np.unique(y)) == 2 else "weighted"
                row = {
                    "Model": name,
                    "Accuracy": round(accuracy_score(y_test, preds), 4),
                    "F1-Score": round(f1_score(y_test, preds, average=avg, zero_division=0), 4),
                    "CV Mean F1": round(cv_scores.mean(), 4),
                    "CV Std": round(cv_scores.std(), 4),
                }
            else:
                row = {
                    "Model": name,
                    "R² Score": round(r2_score(y_test, preds), 4),
                    "RMSE": round(np.sqrt(mean_squared_error(y_test, preds)), 4),
                    "CV Mean R²": round(cv_scores.mean(), 4),
                    "CV Std": round(cv_scores.std(), 4),
                }
            results.append(row)
            trained_models[name] = (model, X_test, y_test)
        except Exception:
            continue

    sort_col = "CV Mean F1" if problem_type == "classification" else "CV Mean R²"
    df_results = pd.DataFrame(results).sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df_results, trained_models


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_).flatten()[:len(feature_names)]
    else:
        return None
    df = pd.DataFrame({"Feature": feature_names, "Importance": imp})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def get_confusion_matrix(model, X_test, y_test, label_encoder=None):
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    labels = label_encoder.classes_ if label_encoder else np.unique(y_test)
    return cm, labels


def tune_best_model(model_name: str, model, X, y, problem_type: str):
    if model_name not in TUNING_GRIDS:
        return model, {}

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if problem_type == "classification" else None
    )
    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    search = RandomizedSearchCV(
        model, TUNING_GRIDS[model_name],
        n_iter=10, cv=3, scoring=scoring,
        random_state=42, n_jobs=-1
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def export_model(model) -> bytes:
    return pickle.dumps(model)
