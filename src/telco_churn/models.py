"""Prétraitement et catalogue des 9 classifieurs du sujet."""
from __future__ import annotations

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config

# Familles pour lesquelles le projet calcule des valeurs SHAP exactes (motifs dans l'API et la démo)
EXPLAINABLE = {"logistic_regression", "xgboost", "lightgbm", "catboost"}


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("scale", StandardScaler())])
    # TotalCharges est très asymétrique : passage au log avant standardisation
    total = Pipeline([("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                      ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, ["tenure", "MonthlyCharges"]),
            ("total", total, ["TotalCharges"]),
            ("bin", "passthrough", config.BINARY),
            ("cat", categorical, config.CATEGORICAL),
        ],
        verbose_feature_names_out=False,
    )


def candidate_models(rs: int = config.RANDOM_STATE) -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, C=0.5),
        "svm": SVC(kernel="rbf", C=1.0, probability=True, random_state=rs),
        "knn": KNeighborsClassifier(n_neighbors=75, weights="distance"),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=10, n_jobs=-1, random_state=rs),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                                        subsample=0.9, random_state=rs),
        "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.9,
                                 colsample_bytree=0.8, min_child_weight=5, eval_metric="auc", n_jobs=-1,
                                 random_state=rs),
        "lightgbm": LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                                   subsample=0.9, subsample_freq=1, colsample_bytree=0.8, random_state=rs,
                                   verbose=-1),
        "catboost": CatBoostClassifier(iterations=500, learning_rate=0.03, depth=4, random_seed=rs, verbose=0,
                                       thread_count=-1, allow_writing_files=False),
    }


def build_pipeline(model) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", model)])
