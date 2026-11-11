"""Explications individuelles exactes (valeurs SHAP) et motifs lisibles.

- Régression logistique : SHAP linéaire exact, coef × (x − moyenne d'entraînement).
- XGBoost / LightGBM / CatBoost : TreeSHAP natif de chaque bibliothèque.
Contributions en log-odds : base + somme des contributions = logit(probabilité de churn).
Les colonnes du prétraitement (one-hot) sont regroupées sur les variables d'origine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS_FR = {
    "tenure": "Ancienneté (mois)",
    "MonthlyCharges": "Facture mensuelle",
    "TotalCharges": "Total facturé",
    "SeniorCitizen": "Client senior",
    "gender": "Genre",
    "Partner": "En couple",
    "Dependents": "Personnes à charge",
    "PhoneService": "Service téléphonique",
    "MultipleLines": "Lignes multiples",
    "InternetService": "Type d'accès Internet",
    "OnlineSecurity": "Sécurité en ligne",
    "OnlineBackup": "Sauvegarde en ligne",
    "DeviceProtection": "Protection de l'appareil",
    "TechSupport": "Support technique",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming films",
    "Contract": "Type de contrat",
    "PaperlessBilling": "Facture dématérialisée",
    "PaymentMethod": "Moyen de paiement",
}


def _feature_map(names, raw, categorical):
    cats = sorted(categorical, key=len, reverse=True)
    return [n if n in raw else next(c for c in cats if n.startswith(c + "_")) for n in names]


def attach_background(pipeline, X_train: pd.DataFrame) -> None:
    """Mémorise la moyenne des variables transformées (référence du SHAP linéaire)."""
    pipeline.shap_background_mean_ = np.asarray(pipeline.named_steps["prep"].transform(X_train)).mean(axis=0)


def shap_values(pipeline, X: pd.DataFrame, categorical: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    Xt = np.asarray(prep.transform(X), dtype=float)
    names = list(prep.get_feature_names_out())
    kind = type(model).__name__
    if kind == "LogisticRegression":
        mean = pipeline.shap_background_mean_
        coef = model.coef_[0]
        contrib = (Xt - mean) * coef
        base = np.full(len(X), float(model.intercept_[0] + mean @ coef))
    elif kind == "LGBMClassifier":
        out = np.asarray(model.predict(Xt, pred_contrib=True))
        contrib, base = out[:, :-1], out[:, -1]
    elif kind == "XGBClassifier":
        import xgboost as xgb
        out = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
        contrib, base = out[:, :-1], out[:, -1]
    elif kind == "CatBoostClassifier":
        from catboost import Pool
        out = model.get_feature_importance(Pool(Xt), type="ShapValues")
        contrib, base = out[:, :-1], out[:, -1]
    else:
        raise TypeError(f"Explications exactes non disponibles pour {kind}")
    mapping = _feature_map(names, list(X.columns), categorical)
    agg = pd.DataFrame(contrib, columns=mapping).T.groupby(level=0, sort=False).sum().T
    return agg[list(X.columns)], base


def reason_codes(X: pd.DataFrame, sv: pd.DataFrame, n_up: int = 3, n_down: int = 2) -> list[dict]:
    out = []
    for i in range(len(X)):
        row = sv.iloc[i].sort_values()

        def item(f):
            v = X.iloc[i][f]
            v = None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)
            return {"feature": f, "label": LABELS_FR.get(f, f), "value": v, "impact": round(float(row[f]), 4)}

        out.append({
            "risk_factors": [item(f) for f in row.index[::-1][:n_up] if row[f] > 0],
            "protective_factors": [item(f) for f in row.index[:n_down] if row[f] < 0],
        })
    return out
